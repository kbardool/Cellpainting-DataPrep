from IPython.display import display, SVG
import os, time
import csv, json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import repeat
from multiprocessing import Pool, process

from scipy import sparse

import rdkit
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import inchi
from rdkit.Chem import INCHI_AVAILABLE

# from rdkit import Chem
from rdkit import RDConfig

from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

import rdkit.Avalon.pyAvalonTools as pat
import rdkit.Chem.MolStandardize as rdms
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.SigFactory import SigFactory    
    
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True  #< set this to False if you want PNGs instead of SVGs

if INCHI_AVAILABLE:
  from rdkit.Chem import InchiReadWriteError
  from rdkit.Chem import MolToInchi, MolBlockToInchi, MolFromInchi, InchiToInchiKey, MolToInchiKey, MolFromMolFile, MolToMolFile
else:
    print('INCHI not available')

from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils

def printMol(mol, width = 450, height = 150):
    bits = AllChem.GetMorganFingerprint(mol, 3) 
    print ('Chembl-id:    %s' % list(mol.GetPropNames()))
#     print ('Chembl-id:    %s' % mol.GetProp('chembl_id'))
    print ('Num of Atoms: %d' % mol.GetNumAtoms())
    print ("#bits:        %d" % len(bits.GetNonzeroElements()))
    print ('Smiles:       %s' % Chem.MolToSmiles(mol, isomericSmiles=True))
    print ("")
    drawMol(mol, width, height)

    
def drawMol(mol, width = 450, height = 150):
#     molSize = (width, height)
    mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg.replace('svg:','')))  
    
    
def saveFingerprints(results, filename):
  """ results is a dictionary: chembl_id -> [fingerprints] """
  with open(filename, 'w') as csvfile:
    fpwriter = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_NONE)
    fpwriter.writerow(["compound","feature"])
    for compound in results:
        for feature in results[compound]:
            fpwriter.writerow( [compound, feature] )
    print(f" => fingerprints written to {filename} ")   
    
    
def saveFingerprintsNpy(results, filename, cmpdlist, ecfp_fold=32000):
    """ write fingerprints to npy, compunds to CSV file """
    ## fp2 : list of tuples : each tuple: (compound id, (array of features, array of counts))
    fp2 =[(x,(np.array(list(results[x].keys())), np.array(list(results[x].values())))) for x in results] #TODO: Modify to save compound list
    
    ## cmpd: tuple of strings, each representing a chembl compound
    ## ecfp: tuple of tuples each tuple being an array of (features, counts)
    cmpd, ecfp   = zip(*fp2) 
    
    ## feat  : tuple of ndarrays, each member being a ndarray of features 
    ## counts: tuple of ndarrays, each member being a ndarray of counts 
    feat, counts = zip(*ecfp)
    
    ## Create indicies for csr matrix
    ## the column indices for row i are stored in indices[indptr[i]:indptr[i+1]] 
    ## their corresponding values are stored in data[indptr[i]:indptr[i+1]]
    lens    = np.array([len(f) for f in feat])
    indptr  = np.concatenate([[0], np.cumsum(lens)])
    indices = np.concatenate(feat) % ecfp_fold
    data    = np.ones(indices.shape[0])
    
    ## Create CSR matrix
    csr     = csr_matrix((data, indices, indptr), shape=(len(feat), ecfp_fold))
    csr.sum_duplicates()
    csr.data[:] = 1.0
    
    # write fingerprint data to numpy CSR sparse matrix 
    np.save(filename, csr)
    print(f" => fingerprints written to {filename} ")    
    
#     # write compounds to csv file 
#     pd.DataFrame(cmpd).to_csv(cmpdlist, header=False)
#     print(f" => compounds written to {cmpdlist} ")       


def my_enumerator(start=0):
    n = start
    while True:
        yield n
        n += 1


def apply_args_and_kwargs(fn, args, kwargs):
    """
    setup process using passed function, args and keyword args
    """
    output = f"\n ==> entering apply_args_and_kwargs for part {args[0]} \n" \
             f" fn:      {type(fn)}    - {fn} \n"\
             f" args:    {type(args)}  - {len(args)}:  {args[0]} {type(args[1])} \n"\
             f" kwargs:  {type(kwargs)}- {type(kwargs)}  {kwargs} \n"\
             f" <== leaving apply_args_and_kwargs \n"
    print(output, flush=True)
    return fn(*args, **kwargs)

    
def starmap_with_kwargs( fn, args , kwargs , pool = None):
    """
    Submit processes using pool.starmap and return 
    """
    print(f" entering starmap_with_kwargs - pool : {pool}")
    print(f" fn: {fn}")
    print(f" args_iter: {type(args)}")
    print(f" kwargs_iter: {type(kwargs)}")

    args_for_starmap = zip(repeat(fn), args , repeat(kwargs))    
    results = pool.starmap(apply_args_and_kwargs, args_for_starmap)
 
    for i in results:
        print(f"  results {i[0]} -     i[1]: {type(i[1])}  {i[1].shape}")
    return results    
    
    
def starmap_with_kwargs_async( fn, args , kwargs , processes = 0):
    output = f" starmap_with_kwargs_async \n"\
             f" processes  : {processes}\n" \
             f" fn         : {fn}  \n" \
             f" args_iter  : {type(args)}   \n" \
             f" kwargs_iter: {type(kwargs)} \n"
    print(output, flush=True)
    
    print(f" {datetime.now().strftime('%X.%f')} | starmap_with_kwargs_async() | Started ")    
    
    args_for_starmap = zip(repeat(fn), args , repeat(kwargs))    
    pool = Pool(processes=processes)
    
    start_time = time.perf_counter()
    # with Pool(processes=processes) as pool:  
    results = pool.starmap_async(apply_args_and_kwargs, args_for_starmap)
    
#     results.wait()
#     if results.ready():
#         r = results.get()
#         finish_time = time.perf_counter()
#         print(f" {datetime.now().strftime('%X.%f')} | starmap_with_kwargs_async() | Results ready. . .  ")    
#         print(f" {datetime.now().strftime('%X.%f')} | All processes complete")
#         print(f" {datetime.now().strftime('%X.%f')} | Program finished in {finish_time-start_time:5f} seconds")
#     else :
#         print('Not Done Yet')
#     print()    
#     print(f" {datetime.now().strftime('%X.%f')} | starmap_with_kwargs_async() | Ended ")    

    return results


def fingerprint_to_sparse_matrix(df, start, end, ncols = 1032):
    indptr  = np.zeros((1,),dtype=np.int8)
    indices = np.empty((0,),dtype=np.int8)
    data    = np.empty((0,),dtype=np.int8)
    ctr = 0
#     FP_ctr, NoFP_ctr = 0,0
#     chunk_id = 0 
#     print(indptr)
#     print(indices)
#     print(data)

    for comp in df[start:end].itertuples():
#         if ctr >= end: break
#         print()
#         if ctr % 1 == 0:
#             print(f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | {ctr} rows processed | index: {comp.Index} | {df.at[comp.Index, 'JCP2022']}", flush = True)
        ctr +=1

#         if comp.pharmacophore_fp is None:
#             print(f" ctr: {ctr} index: {comp.Index}   InChIKey: {comp.InChIKey } \t phramacophore_fp not found   - continue", flush = True)
#             fingerprints.append((df.at[comp.Index, 'JCP2022'], np.nan))
#             NoFP_ctr +=1
#         else:
#             FP_ctr += 1 

        ## Build sparse matrix representation 

        onbits = list(comp.pharmacophore_fp.GetOnBits())
        len_onbits = len(onbits)            
        next_indptr = indptr[-1] + len_onbits
        indptr  = np.concatenate([indptr, [next_indptr]])
        data    = np.concatenate([data, np.ones(len_onbits)])
        indices = np.concatenate([indices,onbits])
#         print()
#         print(f" onbits {len(onbits):5d} : {onbits}")
#         print(f" indptr {len(indptr):5d} : {indptr}")
#         print(f" data   {  len(data):5d} : {data}")
#         print(f" indices{len(indices):5d}: {indices}")
    csr = csr_matrix((data, indices, indptr), shape=(len(indptr)-1, ncols), dtype = np.int8)
#     output = f"\n" \
#              f" ------------------------------------------------------------------------------ \n" \
#              f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | Processing results \n" \
#              f" ------------------------------------------------------------------------------ \n" \
#              f"                                          Total Read : {ctr:6d}      \n" \
#              f" rows with No fingerprint computed                   : {NoFP_ctr:6d} \n" \
#              f" rows with with valid Mol                            : {FP_ctr:6d}   \n" \
#              f"                                               Total : {FP_ctr + NoFP_ctr:6d}\n" \
#              f" ------------------------------------------------------------------------------ \n" \
#              f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | Process end        \n" \
#              f" ------------------------------------------------------------------------------ \n"
    
    return csr , ctr


def sparse_to_dense(csr_mat: sparse._csr.csr_matrix)  -> np.matrix:
    return csr_mat.todense()


def getMorganFingerprint(df: pd.DataFrame, limit:int = 5, nMorgan = 3):
    """
    Get Morgan Fingerprint using standard inchi
    function for looping over all molecules 
    """
    
    ctr =  0
    for  comp in df.itertuples():
        ctr += 1
        if ctr > limit: 
            break
        if  comp.standard_inchi is None: 
            print(' Nan row encounntered - continue')
            continue
        else:
            mol1 =inchi.MolFromInchi(comp.standard_inchi)
            fp1 = AllChem.GetMorganFingerprint(mol1, nMorgan).GetNonzeroElements()
            print(f" ctr: {ctr}  idx: {comp.Index}   inchikey: {comp.InChIKey}  chemblId is : {comp.chembl_id}" )
            yield (comp.chembl_id, fp1)    
            
            
def getMorganFingerprintAsBitVect(df: pd.DataFrame, limit:int = 5, nMorgan = 3):
    """
    Get Morgan Fingerprint using standard inchi as a Bit Vector
    function for looping over all molecules 
    """
    
    ctr =  0
    df_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x,3) for x in df['ROMol'] ]
    for  comp in df.itertuples():
        ctr += 1
        if ctr > limit: 
            break
        if  comp.standard_inchi is None: 
            print(' Nan row encounntered - continue')
            continue
        else:
            mol1 =inchi.MolFromInchi(comp.standard_inchi)
            fp1 = AllChem.GetMorganFingerprint(mol1, nMorgan).GetNonzeroElements()
            print(f" ctr: {ctr}  idx: {comp.Index}   inchikey: {comp.InChIKey}  chemblId is : {comp.chembl_id}" )
            yield (comp.chembl_id, fp1)            
            

### Iterate over dataframe and get standard_inchi, chemblid, 
### and chembl descriptors for each inchikey using chembl_webresource_client
            
def appendInchiFromCWC(df: pd.DataFrame, limit:int = 5, nMorgan:int = 3, chemblDesc = False):
    """ 
    appendInchiFromCWC: 
    
    Iterate over dataframe and get standard_inchi, chemblid, and chembl descriptors 
    for each inchikey using chembl_webresource_client
    """
    NoInchiKey_ctr = 0
    NoInchi_ctr = 0 
    Found_ctr = 0
    ctr = 0 
    chemblDescriptors = [] 
    molecule = new_client.molecule
    
    if 'chembl_id' not in df.columns:
        df['chembl_id'] = None
        print(' Add chembl_id column')
    else:
        print(' chembl_id column exists ')

    if 'standard_inchi' not in df.columns:
        df['standard_inchi'] = None
        print(' Add standard_inchi column')
    else:
        print(' standard_inchi column exists ')
    
    for comp in df.itertuples():
        ctr +=1
        if ctr > limit: 
            break
        if comp is None: 
            print(f" ctr: {ctr}   row is None  - continue")
            continue

        if pd.isna(comp.InChIKey): 
            print(f" ctr: {ctr}   idx: {comp.Index}   inchikey: {comp.InChIKey } \t Nan row encounntered - continue")
            NoInchiKey_ctr += 1
            continue

        mol = molecule.filter(molecule_structures__standard_inchi_key=comp.InChIKey).only(['molecule_chembl_id', 'molecule_structures']) 
        if len(mol) == 0:
            print(f" ctr: {ctr}   idx: {comp.Index}   inchikey: {comp.InChIKey } \t No row found for inchikey on Compound Structures - continue")
            NoInchi_ctr +=1
            continue

        Found_ctr +=1
        
        df.at[comp.Index,'chembl_id'] = mol[0]['molecule_chembl_id']
        df.at[comp.Index,'standard_inchi'] =  mol[0]['molecule_structures']['standard_inchi']
        if chemblDesc:
            descriptors = json.loads(utils.chemblDescriptors(mol[0]['molecule_structures']['molfile']))[0]
            chemblDesc['InChIKey'] = comp.InChIKey
            chemblDescriptors.append(descriptors)
        
#         df_sql = pd.read_sql(f"SELECT  standard_inchi, chembl_id  FROM  chembl_32.compound_structures as a, " \
#                                 "      chembl_32.molecule_dictionary as b  " \
#                                f"where standard_inchi_key = '{comp.InChIKey}' and a.molregno = b.molregno", con = conn)
#       print(df_sql.shape, ' -' , comp['InChIKey'])
#         mol1 =inchi.MolFromInchi(mol[0]['molecule_structures']['standard_inchi'])
#         fp1 = AllChem.GetMorganFingerprint(mol1, nMorgan).GetNonzeroElements()        
        print(f" index: {comp.Index}  inchikey: {comp.InChIKey} - chemblId: {mol[0]['molecule_chembl_id']} Inchi: {(mol[0]['molecule_structures']['standard_inchi'][:50])}....  ")
        
    print('\n')    
    print(f" rows with missing InChiKey in input                 : {NoInchiKey_ctr}")
    print(f" rows with missing InChi on Compound Structure table : {NoInchi_ctr}")
    print(f" rows with with valid Chembl_id and InChi            : {Found_ctr} \n")
    print(f"                                               Total : {NoInchiKey_ctr + NoInchi_ctr + Found_ctr} \n\n") 
    return chemblDescriptors

### `appendFromPubChem` : Iterate over dataframe and get standard_inchi, 
###  chemblid, and chembl descriptors for each inchikey using `PubChemPy`

def getPharamcophoresFromPubChem(df: pd.DataFrame, limit:int = -1, nMorgan:int = 3, chemblDesc = False) -> list:
    from rdkit.Chem.Pharm2D.SigFactory import SigFactory    
    
    """ function for looping over all molecules - for the small Metadata files"""
    NoInchiKey_ctr = 0
    NoMol_ctr = 0 
    NoCompound_ctr = 0
    Found_ctr = 0
    ctr = 0 
    limit = len(df) if limit == -1 else limit
    fingerprints = [] 

    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    print(f"fdefName : {fdefName}")
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3)
    sigFactory.SetBins( [(0,3),(3,8)])
    sigFactory.Init()
    sigFactory.GetSigSize()    
    
    if 'cid' not in df.columns:
        df['cid'] = None
        print(' Add cid column')
    else:
        print(' cid column exists ')

    if 'standard_inchi' not in df.columns:
        df['standard_inchi'] = None
        print(' Add standard_inchi column')
    else:
        print(' standard_inchi column exists ')

    if 'pharmacophore_fp' not in df.columns:
        df['pharmacophore_fp'] = None
        print(' Add pharmacophore_fp column')
    else:
        print(' pharmacophore_fp column exists ')
    
        
    for comp in df.itertuples():
        ctr +=1
        if ctr > limit: 
            break
        if comp is None: 
            print(f" ctr: {ctr} index: {comp.Index}   row is None  - continue")
            continue

        if pd.isna(comp.InChIKey): 
            print(f" ctr: {ctr} index: {comp.Index}   inchikey: {comp.InChIKey } \t Nan row encounntered - continue")
            NoInchiKey_ctr += 1
            continue
 
        cmp = pcp.get_compounds(comp.InChIKey, namespace=u'inchikey', searchtype=None, as_dataframe=False)
 
        if len(cmp) == 0 :
            print(f" ctr: {ctr} index: {comp.Index}   broad_sample: {comp.broad_sample}   InChIKey: {comp.InChIKey } \t No compound returned for inchikey from PubChem get_compounds() - continue")
            NoCompound_ctr +=1
            continue
        else:            
            ## inchi_str = cmp[0].inchi
            df.at[comp.Index,'standard_inchi'] =  cmp[0].inchi
            df.at[comp.Index, 'cid'] = cmp[0].cid
            mol = inchi.MolFromInchi(cmp[0].inchi, sanitize = True, removeHs = True)
            
        if mol is None:
            print(f" ctr: {ctr} index: {comp.Index}   CID: {cmp[0].cid}   InChIKey: {comp.InChIKey } \t No MOL returned for inchi from RDKit - continue")
            fingerprints.append((df.at[comp.Index,'broad_sample'], np.nan))
            NoMol_ctr +=1
            continue
        else:
            Found_ctr +=1
    #       df.at[comp.Index,'chembl_id'] = mol[0]['molecule_chembl_id']        
            fp = Generate.Gen2DFingerprint(mol,sigFactory)
            df.at[comp.Index,'pharmacophore_fp'] = fp    
            fingerprints.append((df.at[comp.Index,'broad_sample'], fp))
            print(f" ctr {ctr} index: {comp.Index}  inchikey: {comp.InChIKey} - CID: {cmp[0].cid:10d} Inchi: {(cmp[0].inchi[:50])}....  ")
    
    
    print('\n')    
    print(f" rows with missing InChiKey in input                 : {NoInchiKey_ctr}")
    print(f" rows with no compound returned from PubChem         : {NoCompound_ctr}")
    print(f" rows with no MOL object returned from RDKit         : {NoMol_ctr}")
    print(f" rows with with valid Chembl_id and InChi            : {Found_ctr} \n")
    print(f"                                               Total : {NoInchiKey_ctr + NoCompound_ctr + NoMol_ctr + Found_ctr} \n\n") 
    return fingerprints


### `appendFromPubChemV2` : Iterate over dataframe and get standard_inchi, chemblid, and chembl descriptors for each inchikey using `PubChemPy`


def appendMolFromRDKit(df: pd.DataFrame, limit:int = -1, verbose = False) -> list:
    
    """ function for looping over all molecules
        for the LARGE metadata file 
    """
    mol_col_name = 'ROMol'

    NoInchiKey_ctr, NoMol_ctr, Mol_Found_ctr = 0, 0, 0
    ctr = 0 
    
    limit = len(df) if limit == -1 else limit
    
    if mol_col_name not in df.columns:
        df[mol_col_name] = None
        print(f' Add {mol_col_name} column')
    else:
        print(f' {mol_col_name} column exists ')
    
        
    for comp in df.itertuples():
        if ctr >= limit: break
        if ctr % 250 == 0:
            print(f" {datetime.now().strftime('%X.%f')} | {ctr} rows processed ")
        ctr +=1

        if pd.isna(comp.InChIKey): 
            print(f" ctr: {ctr} index: {comp.JCP2022}   inchikey: {comp.InChIKey } \t Nan row encounntered - continue")
            NoInchiKey_ctr += 1
            continue

        mol_tmp = inchi.MolFromInchi(comp.InChI, sanitize = True, removeHs = True)
        
#         if verbose:
#             print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey[:10]}  mol: {mol_tmp}")
            
        if mol_tmp is None:
            print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey } \t No MOL returned for inchi from RDKit - continue")
            NoMol_ctr +=1
            continue
        else:           
            df.at[comp.Index, mol_col_name] = mol_tmp
            Mol_Found_ctr +=1

    
    output = \
            f"\n" \
            f" ------------------------------------------------------------------------- \n" \
            f" Results of processing   \n" \
            f" ------------------------------------------------------------------------- \n" \
            f"                                          Total Input: {ctr:6d}\n"\
            f" rows with missing InChiKey in input                 : {NoInchiKey_ctr:6d}  \n" \
            f" rows with no MOL object returned from RDKit         : {NoMol_ctr:6d}       \n" \
            f" rows with valid MOL                                 : {Mol_Found_ctr:6d}   \n" \
            f" ------------------------------------------------------------------------- \n\n" 
    print(output)
    return 0


# def appendDescriptorFromRDKit(df: pd.DataFrame, limit:int = -1, verbose = False) -> list:
#     from rdkit.Chem.Pharm2D.SigFactory import SigFactory    
    
#     """ function for looping over all molecules
#         for the LARGE metadata file
#     """
#     descriptor_col_name = 'TPSA'
#     Descriptor_NotFound = 0 
#     Descriptor_Found = 0 
#     Mol_NA = 0 
#     ctr = 0 
    
#     limit = len(df) if limit == -1 else limit
    
#     if descriptor_col_name not in df.columns:
#         df[descriptor_col_name] = None
#         print(f' Add {descriptor_col_name} column')
#     else:
#         print(f' {descriptor_col_name} column exists ')    
        
    
#     for comp in df.itertuples():
#         if ctr >= limit: break
#         if ctr % 250 == 0:
#             print(f" {datetime.now().strftime('%X.%f')} | {ctr} rows processed ")
#         ctr +=1
            
 
#         if pd.isna(comp.mol):            
#             print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey} - Mol is none")
#             Mol_NA += 1
#             continue 
            
#         tpsa = np.round(Descriptors.TPSA(comp.mol),3)
        
#         if verbose:
#             print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey[:10]}  Descriptor TPSA: {tpsa}")
            
#         if tpsa is None:
#             print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey } \t No MOL returned for inchi from RDKit - continue")
#             tpsa = np.nan
#             Descriptor_NotFound +=1
#             continue
#         else:
#             Descriptor_Found +=1
        
#         df.at[comp.Index, descriptor_col_name] = tpsa
    
#     output = \
#             f"\n" \
#             f" ------------------------------------------------------------------------- \n" \
#             f" Results of processing   \n" \
#             f" ------------------------------------------------------------------------- \n" \
#             f"                               Total Input: {ctr:6d}\n"\
#             f" rows with No Mol                         : {Mol_NA:6d}   \n" \
#             f" rows with valid Descriptor               : {Descriptor_Found :6d}   \n" \
#             f" rows with invalid Descriptor             : {Descriptor_NotFound:6d} \n" \
#             f" -------------------------------------------------------------------------- \n\n" 
#     print(output)
#     return df 


def getSmilesFromRDKit(df: pd.DataFrame, limit:int = -1, append_col = False,  verbose = False) -> pd.DataFrame:
    # from rdkit.Chem.Pharm2D.SigFactory import SigFactory    
    
    """ function for looping over all molecules
        for the LARGE metadata file 
    """
    smiles_col_name = 'smiles'
    smiles_list = []
    NoInchiKey_ctr = 0
    NoMol_ctr, Mol_Found_ctr = 0, 0
    NoSmiles_ctr, Smiles_Found_ctr = 0, 0
    ctr = 0 
    
    limit = len(df) if limit == -1 else limit
    
    if append_col and smiles_col_name not in df.columns:
        df[smiles_col_name] = np.nan
        print(f' Add {smiles_col_name} column')
    else:
        print(f' {smiles_col_name} column exists')    
        
    for comp in df.itertuples():
        if ctr >= limit: break
        if ctr % 250 == 0:
            print(f" {datetime.now().strftime('%X.%f')} | {ctr} rows processed ")
        ctr +=1

        canon_smiles = None
        mol_tmp = None
        
        if pd.isna(comp.Metadata_InChIKey): 
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   inchikey: {comp.Metadata_InChIKey } \t Nan row encounntered - continue")
            NoInchiKey_ctr += 1
            continue

        mol_tmp = inchi.MolFromInchi(comp.Metadata_InChI, sanitize = True, removeHs = True)
        
        if verbose:
            print(f" ctr: {ctr} index: {comp.JCP2022}   InChIKey: {comp.InChIKey[:10]}  mol: {mol_tmp}")
            
        if mol_tmp is None:
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   InChIKey: {comp.Metadata_InChIKey } \t No MOL returned for inchi from RDKit - continue")
            NoMol_ctr +=1
            continue
        else:
            Mol_Found_ctr +=1
            
        canon_smiles = pat.GetCanonSmiles(mol_tmp)
        if canon_smiles is None:
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   InChIKey: {comp.Metadata_InChIKey } \t No SMILES returned for mol from RDKit - continue")                
            NoSmiles_ctr +=1
            continue
        else :
            Smiles_Found_ctr += 1
            
        smiles_list.append((df.at[comp.Index,'Metadata_JCP2022'], canon_smiles))
        
        if append_col:     
            df.at[comp.Index, smiles_col_name] = canon_smiles

    df_smiles = pd.DataFrame(smiles_list, columns=['JCP2022', smiles_col_name ])
    output = \
            f"\n" \
            f" ------------------------------------------------------------------------- \n" \
            f" Results of processing   \n" \
            f" ------------------------------------------------------------------------- \n" \
            f"                                          Total Input: {ctr:6d}\n"\
            f" rows with missing InChiKey in input                 : {NoInchiKey_ctr:6d}  \n" \
            f" rows with no MOL object returned from RDKit         : {NoMol_ctr:6d}       \n" \
            f" rows with valid MOL                                 : {Mol_Found_ctr:6d}   \n" \
            f" rows with no Smiles object returned from RDKit      : {NoSmiles_ctr:6d}    \n" \
            f" rows with with valid Chembl_id and InChi            : {Smiles_Found_ctr:6d}\n" \
            f" ------------------------------------------------------------------------- \n\n" 
    print(output)
    return df_smiles


def getTPSAFromRDKit(df: pd.DataFrame, limit:int = -1, append_col=False,  verbose = False) -> pd.DataFrame:
    # from rdkit.Chem.Pharm2D.SigFactory import SigFactory    
    
    """ function for looping over all molecules
        for the LARGE metadata file 
    """
    tpsa_col_name = 'TPSA'
    tpsa_list = []
    NoInchiKey_ctr = 0
    NoMol_ctr, Mol_Found_ctr = 0, 0
    NoTPSA_ctr, TPSA_Found_ctr = 0, 0
    ctr = 0 
    print(f"getTPSAFromRDKit() input parmL limit: {limit} \t append_column :  {append_col} \t verbose: {verbose}")    
    
    limit = len(df) if limit == -1 else limit
    if append_col and tpsa_col_name not in df.columns:
        df[tpsa_col_name] = None
        print(f' Add {tpsa_col_name} column')
    else:
        print(f' {tpsa_col_name} column exists')    
        
    for comp in df.itertuples():
        if ctr >= limit: break
        if ctr % 250 == 0:
            print(f" {datetime.now().strftime('%X.%f')} | {ctr} rows processed ")
        ctr +=1

        mol_tpsa = None
        mol_tmp = None
        if pd.isna(comp.Metadata_InChIKey): 
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   inchikey: {comp.Metadata_InChIKey } \t Nan row encounntered - continue")
            NoInchiKey_ctr += 1
        else:
            mol_tmp = inchi.MolFromInchi(comp.Metadata_InChI, sanitize = True, removeHs = True)
            if verbose:
                print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   InChIKey: {comp.Metadata_InChIKey[:10]}  mol: {mol_tmp}")
            
        if mol_tmp is None:
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   InChIKey: {comp.Metadata_InChIKey } \t No MOL returned for inchi from RDKit - continue")
            NoMol_ctr +=1
            continue
        else:
            Mol_Found_ctr +=1
            mol_tpsa = np.round(Descriptors.TPSA(mol_tmp),3)
        
        if mol_tpsa is None:
            print(f" ctr: {ctr} index: {comp.Metadata_JCP2022}   InChIKey: {comp.Metadata_InChIKey } \t No TPSA returned for mol from RDKit - continue")                
            NoTPSA_ctr +=1
        else :
            TPSA_Found_ctr += 1
            
            
        tpsa_list.append((df.at[comp.Index,'Metadata_JCP2022'], mol_tpsa))           
        
        if append_col:
            df.at[comp.Index, tpsa_col_name] = mol_tpsa

    df_tpsa = pd.DataFrame(tpsa_list, columns=['JCP2022', tpsa_col_name ])
    output = \
            f"\n" \
            f" ------------------------------------------------------------------------- \n" \
            f" Results of processing   \n" \
            f" ------------------------------------------------------------------------- \n" \
            f"                                          Total Input: {ctr:6d}\n"\
            f" rows with missing InChiKey in input                 : {NoInchiKey_ctr:6d}  \n" \
            f" rows with no MOL object returned from RDKit         : {NoMol_ctr:6d}       \n" \
            f" rows with valid MOL                                 : {Mol_Found_ctr:6d}   \n" \
            f" rows with TPSA not calculated                       : {NoTPSA_ctr:6d}    \n" \
            f" rows with with valid TPSA                           : {TPSA_Found_ctr:6d}\n" \
            f" ------------------------------------------------------------------------- \n\n" 
    print(output)
    return df_tpsa


def getPharmacophoresFromRDKit( chunk_id:int = 0, df_chunk: pd.DataFrame = None, limit:int = -1, 
                                nMorgan:int = 3 , chemblDesc = False, append_pharmacophore = False, progress_int = 250) -> list:
    """
    function for the LARGE metadata file 
    Read each compound from the metadata file entry, Get the RDKit molecule, and compute the 2D pharmacophore

    returns
    --------
    chunk_id:
    df_chunk:
    df_fingerprints
    
    """
    NoInchiKey_ctr = 0
    NoMol_ctr, Mol_Found_ctr = 0, 0
    NoFP_ctr, FP_ctr = 0,0 
    ctr = 0 
    
    limit = len(df_chunk) if limit == -1 else limit
    
    print(f" progress_int : {progress_int}")
    fingerprints = [] 
    pharmacophore_fp_col = 'pharmacophore_fp'
    pharmacophore_base64_col = 'pharmacophore_base64'

    output = f"\n" \
    f" ------------------------------------------------------------------------- \n" \
    f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | process started - chunk sz: {df_chunk.shape[0]}  append_phar: {append_pharmacophore}\n" \
    f" ------------------------------------------------------------------------- \n"
    print(output, flush = True)
    
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3)
    sigFactory.SetBins( [(0,3),(3,8)])
    sigFactory.Init()
    sigFactory.GetSigSize()    
#     print(f" RDKit BaseFeatures.fdef : {fdefName}")

    if append_pharmacophore and pharmacophore_base64_col not in df_chunk.columns:
        df_chunk[pharmacophore_base64_col] = None
        output = f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | Add {pharmacophore_base64_col} column"
    else:
        output = f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | {pharmacophore_base64_col} column exists"
    print(output, flush = True)
        
        
    for comp in df_chunk.itertuples():
        if ctr > limit: break
        if ctr % progress_int == 0:
            print(f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | {ctr} rows processed ", flush = True)
        ctr +=1
 
        fp, mol_tmp = None, None
    
        if pd.isna(comp.InChIKey): 
            print(f" ctr: {ctr} index: {comp.Index}   inchikey: {comp.InChIKey } \t Nan row encounntered - continue", flush = True)
            fingerprints.append((df_chunk.at[comp.Index, 'JCP2022'], np.nan))            
            NoInchiKey_ctr += 1
        else:        
            ## Get Molecule 
            mol_tmp = inchi.MolFromInchi(comp.InChI, sanitize = True, removeHs = True)
 
        if mol_tmp is None:
            print(f" ctr: {ctr} index: {comp.Index}   InChIKey: {comp.InChIKey } \t No MOL returned for inchi from RDKit - continue", flush = True)
            fingerprints.append((df_chunk.at[comp.Index, 'JCP2022'], np.nan))
            NoMol_ctr +=1
#             continue
        else:
            Mol_Found_ctr +=1    
            fp = Generate.Gen2DFingerprint(mol_tmp, sigFactory)
            
        if fp is None:
            print(f" ctr: {ctr} index: {comp.Index}   InChIKey: {comp.InChIKey } \t No Fingerprint computed from molecule file  - continue", flush = True)
            fingerprints.append((df_chunk.at[comp.Index, 'JCP2022'], np.nan))
            NoFP_ctr +=1
#             continue
        else:
            FP_ctr += 1 

        fingerprints.append((df_chunk.at[comp.Index,'JCP2022'], fp))

        if append_pharmacophore:
            fp_base64 = rdkit.DataStructs.cDataStructs.SparseBitVect.ToBase64(fp)
            df_chunk.at[comp.Index, pharmacophore_base64_col] = fp_base64
    
    df_fingerprints = pd.DataFrame(fingerprints, columns=['JCP2022', pharmacophore_fp_col ])        
    
    output = f"\n" \
             f" ------------------------------------------------------------------------- \n" \
             f" Results of processing chunk {chunk_id} \n" \
             f" ------------------------------------------------------------------------- \n" \
             f"                                          Total Read : {ctr:6d}\n" \
             f" rows with missing InChiKey in input                 : {NoInchiKey_ctr:6d} \n" \
             f" rows with no MOL object returned from RDKit         : {NoMol_ctr:6d}      \n" \
             f" rows with with valid Mol                            : {Mol_Found_ctr:6d}   \n" \
             f" rows with No fingerprint computed                   : {NoFP_ctr:6d}       \n" \
             f" rows with with valid Mol                            : {FP_ctr:6d}   \n" \
             f"                                               Total : {NoInchiKey_ctr +  NoMol_ctr + NoFP_ctr:6d}\n" \
             f" ------------------------------------------------------------------------- \n" \
             f" {datetime.now().strftime('%X.%f')} | chunk_id: {chunk_id} | Process end   \n"\
             f" ------------------------------------------------------------------------- \n"
    print(output, flush = True)
    return chunk_id, df_chunk, df_fingerprints



