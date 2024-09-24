from types import  SimpleNamespace

def project_datasets(metadata_path = './metadata/', input_path='./input/', output_path = './output/', prefix = ''):
    ds = SimpleNamespace()
    # metadata_path = "./metadata/"
    # input_path ="./metadata/"
    # output_path ="./output_11102023/"
    # output_path ="./output_04162024/"
    ds.prefix = prefix    ### Target-2' , 'MOA'
    prefix_lc = prefix.lower().replace('-', '_')


    # compoundMetadataInputFile     =  f"{input_path}{prefix_lc}_compound_metadata.csv"
    # compoundMetadataInputFile       = f"{input_path}JUMP-{prefix}_compound_library.csv"
    # compoundPharmacophoreFile       = f"{output_path}{prefix_lc}_compound_pharmacophores_sparse.pkl"
    # compoundMetadataInputFile   = f"{input_path}JUMP-{prefix}_compound_library.csv"

    ds.compoundMetadata                         = f"{metadata_path}{prefix_lc}compound.csv"
    ds.plateMetadata                            = f"{metadata_path}{prefix_lc}plate_new.csv"
    ds.wellMetadata                             = f"{metadata_path}{prefix_lc}well.csv"
    ds.ORFMetadata                              = f"{metadata_path}{prefix_lc}orf.csv"
    ds.CrisprMetadata                           = f"{metadata_path}{prefix_lc}crispr.csv"
    ds.profileMetadataFile                      = f"{input_path}profile_metadata.pkl"
    ds.parquetMetadataFile                      = f"{input_path}parquet_columns.pkl"
    ds.compoundMetadataInputFile                = f"{input_path}{prefix_lc}compound.csv"
    ds.plateMetadataInputFile                   = f"{input_path}{prefix_lc}plate_new.csv"
    ds.wellMetadataInputFile                    = f"{input_path}{prefix_lc}well_new.csv"


    ds.compoundMetadataInputFile                = f"{input_path}{prefix_lc}compound_metadata.csv"
    ds.compoundMetadataSmilesFile               = f"{output_path}{prefix_lc}compound_metadata_smiles.csv"
    ds.compoundMetadataSmilesCleanFile          = f"{output_path}{prefix_lc}compound_metadata_smiles_clean.csv"
    ds.compoundMetadataCleanFile                = f"{output_path}{prefix_lc}compound_metadata_clean.csv"
    ds.compoundMetadataTPSAFile                 = f"{output_path}{prefix_lc}compound_metadata_tpsa.csv"
    ds.compoundMetadataTPSACleanFile            = f"{output_path}{prefix_lc}compound_metadata_tpsa_clean.csv"
    ds.compoundTPSAFile                         = f"{output_path}{prefix_lc}compound_TPSA.csv"
    ds.compoundTPSACleanFile                    = f"{output_path}{prefix_lc}compound_TPSA_clean.csv"

    ds.compoundPharmacophoreFile                = f"{output_path}{prefix_lc}compound_pharmacophores_sparse.pkl"
    ds.compoundPharmacophoreCleanFile           = f"{output_path}{prefix_lc}compound_pharmacophores_sparse_clean.pkl"
    ds.compoundPharmacophoreDenseFile           = f"{output_path}{prefix_lc}compound_pharmacophores_dense.npy"
    ds.compoundPharmacophoreDenseZipFile        = f"{output_path}{prefix_lc}compound_pharmacophores_dense.npz"
    ds.compoundMetadataPharmacophoreFile        = f"{output_path}{prefix_lc}compound_metadata_pcfp.csv"
    ds.compoundMetadataOutputFile_3             = f"{output_path}{prefix_lc}compound_metadata_pcfp.csv"
    
    # ds.compoundPharmacophoreFile              = f"{output_path}{prefix_lc}compound_pcfp_sparse.pkl"
    # ds.compoundPharmacophoreCleanFile         = f"{output_path}{prefix_lc}compound_pcfp_sparse_clean.pkl"
    # ds.compoundPharmacophoreDenseFile         = f"{output_path}{prefix_lc}compound_pcfp_dense.npy"
    # ds.compoundPharmacophoreDenseZipFile      = f"{output_path}{prefix_lc}compound_pcfp_dense.npz"
    
    # ds.CompoundExtendedMetadataFile             = f"{output_path}{prefix_lc}/compound_extended_metadata.csv"
    # ds.CompoundExtendedMetadata5SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_5samples.csv"
    # ds.CompoundExtendedMetadata3SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_3samples.csv"
    # ds.CompoundExtendedMetadata2SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_2samples.csv"

    ds.CompoundExtendedMetadataFile             = f"{output_path}{prefix_lc}compound_extended_metadata.csv"
    ds.CompoundExtendedMetadataMoreThan3File    = f"{output_path}{prefix_lc}compound_extended_metadata_morethan_3_wells.csv"
    ds.CompoundExtendedMetadata5SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_5samples.csv"
    ds.CompoundExtendedMetadata4SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_4samples.csv"
    ds.CompoundExtendedMetadata3SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_3samples.csv"
    ds.CompoundExtendedMetadata2SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_2samples.csv"

    ds.CompoundProfiles2SampleFile         = f"{output_path}{prefix_lc}/2_sample_profiles/2sample_profiles_{{0:03d}}{{1:s}}.csv"
    ds.CompoundProfiles3SampleFile         = f"{output_path}{prefix_lc}/3_sample_profiles/3sample_profiles_{{0:03d}}{{1:s}}.csv"
    ds.CompoundProfiles5SampleFile         = f"{output_path}{prefix_lc}/5_sample_profiles/5sample_profiles_{{0:03d}}{{1:s}}.csv"

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