from types import  SimpleNamespace

def project_datasets(metadata_path = './metadata/', input_path='./input/', output_path = './output/', prefix = ''):
    ds = SimpleNamespace()
    ds.metadata_path = metadata_path
    ds.input_path = input_path
    ds.output_path = output_path
    ds.prefix = prefix    ### Target-2' , 'MOA'
    ds.prefix_lc = prefix.lower().replace('-', '_')


    # compoundMetadataInputFile       =  f"{input_path}{prefix_lc}_compound_metadata.csv"
    # compoundMetadataInputFile       = f"{input_path}JUMP-{prefix}_compound_library.csv"
    # compoundPharmacophoreFile       = f"{output_path}{prefix_lc}_compound_pharmacophores_sparse.pkl"
    # compoundMetadataInputFile       = f"{input_path}JUMP-{prefix}_compound_library.csv"

    ds.compoundMetadata                         = f"{ds.metadata_path}{ds.prefix_lc}compound.csv"
    # ds.compoundMetadataInputFile                = f"{metadata_path}{prefix_lc}compound.csv"
    
    ds.plateMetadata                            = f"{ds.metadata_path}{ds.prefix_lc}plate_new.csv"
    # ds.plateMetadataInputFile                   = f"{metadata_path}{prefix_lc}plate_new.csv"
    
    ds.wellMetadata                             = f"{ds.metadata_path}{ds.prefix_lc}well.csv"
    # ds.wellMetadataInputFile                    = f"{metadata_path}{prefix_lc}well_new.csv"
    
    ds.ORFMetadata                              = f"{ds.metadata_path}{ds.prefix_lc}orf.csv"
    ds.CrisprMetadata                           = f"{ds.metadata_path}{ds.prefix_lc}crispr.csv"
    
    ds.profileMetadataFile                      = f"{ds.metadata_path}{ds.prefix_lc}profile_metadata.pkl"
    ds.parquetMetadataFile                      = f"{ds.metadata_path}{ds.prefix_lc}parquet_columns.pkl"
    

    ds.compoundMetadataCleanFile                = f"{ds.output_path}{ds.prefix_lc}compound_metadata_clean.csv"
    ds.compoundMetadataSmilesFile               = f"{ds.output_path}{ds.prefix_lc}compound_metadata_smiles.csv"
    # ds.compoundMetadataSmilesCleanFile          = f"{ds.output_path}{ds.prefix_lc}compound_metadata_smiles_clean.csv"
    ds.compoundMetadataTPSAFile                 = f"{ds.output_path}{ds.prefix_lc}compound_metadata_tpsa.csv"
    ds.compoundMetadataTPSACleanFile            = f"{ds.output_path}{ds.prefix_lc}compound_metadata_tpsa_clean.csv"
    ds.compoundTPSAFile                         = f"{ds.output_path}{ds.prefix_lc}compound_TPSA.csv"
    ds.compoundTPSACleanFile                    = f"{ds.output_path}{ds.prefix_lc}compound_TPSA_clean.csv"
    ds.compoundMetadataPharmacophoreFile        = f"{ds.output_path}{ds.prefix_lc}compound_metadads.ta_pcfp.csv"

    ds.compoundPharmacophoreSparseFile          = f"{ds.output_path}{ds.prefix_lc}compound_pharmacophores_sparse.pkl"
    ds.compoundPharmacophoreCSVFile             = f"{ds.output_path}{ds.prefix_lc}compound_pharmacophores.csv"
    ds.compoundPharmacophoreNumPyFile           = f"{ds.output_path}{ds.prefix_lc}compound_pharmacophores.npy"
    ds.compoundPharmacophoreNumPyZipFile        = f"{ds.output_path}{ds.prefix_lc}compound_pharmacophores_zip.npz"
 
    
    # ds.compoundPharmacophoreFile              = f"{output_path}{prefix_lc}compound_pcfp_sparse.pkl"
    # ds.compoundPharmacophoreCleanFile         = f"{output_path}{prefix_lc}compound_pcfp_sparse_clean.pkl"
    # ds.compoundPharmacophoreDenseFile         = f"{output_path}{prefix_lc}compound_pcfp_dense.npy"
    # ds.compoundPharmacophoreDenseZipFile      = f"{output_path}{prefix_lc}compound_pcfp_dense.npz"
    
    # ds.CompoundExtendedMetadataFile             = f"{output_path}{prefix_lc}/compound_extended_metadata.csv"
    # ds.CompoundExtendedMetadata5SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_5samples.csv"
    # ds.CompoundExtendedMetadata3SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_3samples.csv"
    # ds.CompoundExtendedMetadata2SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_2samples.csv"

    ds.CompoundExtendedMetadataFile             = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata.csv"
    ds.CompoundExtendedMetadataMoreThan3File    = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata_morethan_3_wells.csv"
    ds.CompoundExtendedMetadata5SampleFile      = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata_5samples.csv"
    ds.CompoundExtendedMetadata4SampleFile      = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata_4samples.csv"
    ds.CompoundExtendedMetadata3SampleFile      = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata_3samples.csv"
    ds.CompoundExtendedMetadata2SampleFile      = f"{ds.output_path}{ds.prefix_lc}compound_extended_metadata_2samples.csv"

    ds.compoundPharmacophore3SampleFile         = f"{ds.output_path}{ds.prefix_lc}compound_pharmacophores_3samples.csv"
    # ds.compoundPharmacophore3SampleNumPyFile    = f"{output_path}{prefix_lc}compound_pharmacophores_3samples.npy"
    # ds.compoundPharmacophore3SampleNumPyZipFile = f"{output_path}{prefix_lc}compound_pharmacophores_dense_zip_3samples.npz"
    
    ds.CompoundProfiles2SampleFile         = f"{ds.output_path}{ds.prefix_lc}/2_sample_profiles/2sample_profiles_{{0:03d}}{{1:s}}.csv"
    ds.CompoundProfiles3SampleFile         = f"{ds.output_path}{ds.prefix_lc}/3_sample_profiles/3sample_profiles_{{0:03d}}{{1:s}}.csv"
    ds.CompoundProfiles5SampleFile         = f"{ds.output_path}{ds.prefix_lc}/5_sample_profiles/5sample_profiles_{{0:03d}}{{1:s}}.csv"

    ## gz, bz2, zip, tar, tar.gz, tar.bz2
    # types = ['.gz', '.bz2','.zip', '.tar', '.tar.gz', '.tar.bz2']
    ds.gzip_compression_options = {"method": "gzip", 'compresslevel': 1,"mtime": 1}
    ds.type_bz2 = 'bz2'
    ds.type_gzip = 'gz'


    ##
    ##
    ##

    ds.root_folder = "/mnt/i/cellpainting-gallery"
    ds.root_folder = "/nvme/cellpainting"
    ds.profile_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )

    ## images
    ds.loaddata_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/load_data_csv/"
        "{Metadata_Batch}/{Metadata_Plate}/load_data_with_illum.parquet"
    )

    # ds.csv_formatter = (
    #     "/mnt/i/cellpainting-gallery/cpg0016-jump/{Metadata_Plate}.csv")
    #     # "{Metadata_Source}/workspace/profiles/"
    #     # "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.csv"

    # ds.local_formatter = (
    #     "/mnt/i/cellpainting-gallery/cpg0016-jump/"
    #     "{Metadata_Source}/workspace/profiles/"
    #     "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    # )

    ds.csv_formatter = (
        "{0}/cpg0016-jump/{2}.csv")
        # "{Metadata_Source}/workspace/profiles/"
        # "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.csv"

    ds.local_formatter = (
        "{0}/cpg0016-jump/"
        "{1}/workspace/profiles/"
        "{2}/{3}/{3}.parquet"
    )

    return ds

def display_project_datasets(ds):
    print(f" Path and Prefixes : ")
    print(f" ------------------- ")
    print(f" prefix                                      : {ds.prefix}" )
    print(f" prefix_lc                                   : {ds.prefix_lc}")
    print(f" metadata path                               : {ds.metadata_path}")
    print(f" input path                                  : {ds.input_path}")
    print(f" output path                                 : {ds.output_path}")
    print()
    print(f" Input & Metadata Datasets : ")
    print(f" --------------------------- ")
    print(f" Compound Metadata Input File                : {ds.compoundMetadata}")
    print(f" Plate Metadata Input File                   : {ds.plateMetadata}")
    print(f" Well Metadata Input File                    : {ds.wellMetadata}")
    print()
    print(f" Profiles Metadata File                      : {ds.profileMetadataFile}")
    print(f" Parquet  Metadata File                      : {ds.parquetMetadataFile}")   
    print()
    print(f" Compound Metadata : ")
    print(f" ------------------- ")
    # print(f" Compound Metadata Cleaned                   : {ds.compoundMetadataCleanFile}")
    print(f" Metadata + Smiles                           : {ds.compoundMetadataSmilesFile}")
    # print(f"          + Smiles Cleaned                   : {ds.compoundMetadataSmilesCleanFile}")
    print(f"          + TPSA                             : {ds.compoundMetadataTPSAFile}")
    print(f"          + TPSA Cleaned (TPSA <> 0)         : {ds.compoundMetadataTPSACleanFile}")
    print(f"          + Base64 Pharmacophores            : {ds.compoundMetadataPharmacophoreFile}")
    print()
    print(f" Compound/TPSA File:")
    print(f" -------------------")
    print(f" compound TPSA File                          : {ds.compoundTPSAFile}")
    print(f" compound TPSA Clean File                    : {ds.compoundTPSACleanFile}\n")
    print()
    print(f" Compound/Pharmacophore Files :")
    print(f" ------------------------------")
    # print(f" compound Pharmacophore Sparse Cleaned File  : {ds.compoundPharmacophoreCleanFile}\n")
    print(f" compound Pharmacophore Sparse File          : {ds.compoundPharmacophoreSparseFile}")
    print(f" compound Pharmacophore CSV File             : {ds.compoundPharmacophoreCSVFile}")
    print(f" compound Pharmacophore Numpy File           : {ds.compoundPharmacophoreNumPyFile}")
    print(f" compound Pharmacophore Numpy Zipped File    : {ds.compoundPharmacophoreNumPyZipFile}")
    print()
    print(f" compound Pharmacophore 3Sample File         : {ds.compoundPharmacophore3SampleFile}")
    # print(f" compound Pharmacophore 3Sample NumPy File  : {ds.compoundPharmacophore3SampleDenseFile}")
    # print(f" compound Pharmacophore 3Sample Dense Zipped File : {ds.compoundPharmacophore3SampleDenseZipFile}")
    print()
    print(f" Compound Extended Metadata files:")
    print(f" ---------------------------------")
    print(f" Compound Extended MetadataFile              : {ds.CompoundExtendedMetadataFile }")
    print()
    print(f" Compound Extended Metadata 5 SampleFile     : {ds.CompoundExtendedMetadata5SampleFile }")
    print(f" Compound Extended Metadata 3 SampleFile     : {ds.CompoundExtendedMetadata3SampleFile }")
    print(f" Compound Extended Metadata 2 SampleFile     : {ds.CompoundExtendedMetadata2SampleFile }")
    print()
    print(f" Compound Profiles files:")
    print(f" ------------------------")
    print(f" Compound Profiles 5 Samples File            : {ds.CompoundProfiles5SampleFile}")
    print(f" Compound Profiles 3 Samples File            : {ds.CompoundProfiles3SampleFile}")
    print(f" Compound Profiles 2 Samples File            : {ds.CompoundProfiles2SampleFile}")
    print()


