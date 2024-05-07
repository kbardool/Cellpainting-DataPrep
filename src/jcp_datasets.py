metadata_path = "./metadata/"
input_path ="./metadata/"
output_path ="./output_11102023/"
# output_path ="./output_04162024/"
prefix = '' ### Target-2' , 'MOA'
prefix_lc = prefix.lower().replace('-', '_')


# compoundMetadataInputFile     =  f"{input_path}{prefix_lc}_compound_metadata.csv"
# compoundMetadataInputFile       = f"{input_path}JUMP-{prefix}_compound_library.csv"

# compoundPharmacophoreFile       = f"{output_path}{prefix_lc}_compound_pharmacophores_sparse.pkl"

# compoundMetadataInputFile   = f"{input_path}JUMP-{prefix}_compound_library.csv"

compoundMetadata                         = f"{metadata_path}{prefix_lc}compound.csv"
plateMetadata                            = f"{metadata_path}{prefix_lc}plate_new.csv"
wellMetadata                             = f"{metadata_path}{prefix_lc}well.csv"
ORFMetadata                              = f"{metadata_path}{prefix_lc}orf.csv"
CrisprMetadata                           = f"{metadata_path}{prefix_lc}crispr.csv"
compoundMetadataInputFile                = f"{input_path}{prefix_lc}compound.csv"
plateMetadataInputFile                   = f"{input_path}{prefix_lc}plate_new.csv"
wellMetadataInputFile                    = f"{input_path}{prefix_lc}well_new.csv"


compoundMetadataInputFile                =  f"{input_path}{prefix_lc}compound_metadata.csv"
 
compoundMetadataSmilesFile               = f"{output_path}{prefix_lc}compound_metadata_smiles.csv"
compoundMetadataSmilesCleanFile          = f"{output_path}{prefix_lc}compound_metadata_smiles_clean.csv"
compoundMetadataCleanFile                = f"{output_path}{prefix_lc}compound_metadata_clean.csv"

compoundMetadataTPSAFile                 = f"{output_path}{prefix_lc}compound_metadata_tpsa.csv"
compoundMetadataTPSACleanFile            = f"{output_path}{prefix_lc}compound_metadata_tpsa_clean.csv"
compoundTPSAFile                         = f"{output_path}{prefix_lc}compound_TPSA.csv"
compoundTPSACleanFile                    = f"{output_path}{prefix_lc}compound_TPSA_clean.csv"

compoundPharmacophoreFile                = f"{output_path}{prefix_lc}compound_pharmacophores_sparse.pkl"
compoundPharmacophoreCleanFile           = f"{output_path}{prefix_lc}compound_pharmacophores_sparse_clean.pkl"
compoundPharmacophoreDenseFile           = f"{output_path}{prefix_lc}compound_pharmacophores_dense.npy"
compoundPharmacophoreDenseZipFile        = f"{output_path}{prefix_lc}compound_pharmacophores_dense.npz"
compoundMetadataPharmacophoreFile        = f"{output_path}{prefix_lc}compound_metadata_pcfp.csv"
compoundMetadataOutputFile_3             = f"{output_path}{prefix_lc}compound_metadata_pcfp.csv"
# compoundPharmacophoreFile              = f"{output_path}{prefix_lc}compound_pcfp_sparse.pkl"
# compoundPharmacophoreCleanFile         = f"{output_path}{prefix_lc}compound_pcfp_sparse_clean.pkl"
# compoundPharmacophoreDenseFile         = f"{output_path}{prefix_lc}compound_pcfp_dense.npy"
# compoundPharmacophoreDenseZipFile      = f"{output_path}{prefix_lc}compound_pcfp_dense.npz"

# CompoundExtendedMetadataFile             = f"{output_path}{prefix_lc}/compound_extended_metadata.csv"
# CompoundExtendedMetadata5SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_5samples.csv"
# CompoundExtendedMetadata3SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_3samples.csv"
# CompoundExtendedMetadata2SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_2samples.csv"

CompoundExtendedMetadataFile             = f"{output_path}{prefix_lc}compound_extended_metadata.csv"
CompoundExtendedMetadataMoreThan3File    = f"{output_path}{prefix_lc}compound_extended_metadata_morethan_3_wells.csv"
CompoundExtendedMetadata5SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_5samples.csv"
CompoundExtendedMetadata4SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_4samples.csv"
CompoundExtendedMetadata3SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_3samples.csv"
CompoundExtendedMetadata2SampleFile      = f"{output_path}{prefix_lc}compound_extended_metadata_2samples.csv"

profileMetadataFile                      = f"{input_path}profile_metadata.pkl"
parquetMetadataFile                      = f"{input_path}parquet_columns.pkl"

CompoundProfiles2SampleFile         = f"{output_path}{prefix_lc}/2_sample_profiles/2sample_profiles_{{0:03d}}{{1:s}}.csv"
CompoundProfiles3SampleFile         = f"{output_path}{prefix_lc}/3_sample_profiles/3sample_profiles_{{0:03d}}{{1:s}}.csv"
CompoundProfiles5SampleFile         = f"{output_path}{prefix_lc}/5_sample_profiles/5sample_profiles_{{0:03d}}{{1:s}}.csv"

## gz, bz2, zip, tar, tar.gz, tar.bz2
# types = ['.gz', '.bz2','.zip', '.tar', '.tar.gz', '.tar.bz2']
gzip_compression_options = {"method": "gzip", 'compresslevel': 1,"mtime": 1}
type_bz2 = 'bz2'
type_gzip = 'gz'


##
##
##

root_folder = "/mnt/i/cellpainting-gallery"
root_folder = "/nvme/cellpainting"
profile_formatter = (
    "s3://cellpainting-gallery/cpg0016-jump/"
    "{Metadata_Source}/workspace/profiles/"
    "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
)

## images
loaddata_formatter = (
    "s3://cellpainting-gallery/cpg0016-jump/"
    "{Metadata_Source}/workspace/load_data_csv/"
    "{Metadata_Batch}/{Metadata_Plate}/load_data_with_illum.parquet"
)

# csv_formatter = (
#     "/mnt/i/cellpainting-gallery/cpg0016-jump/{Metadata_Plate}.csv")
#     # "{Metadata_Source}/workspace/profiles/"
#     # "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.csv"

# local_formatter = (
#     "/mnt/i/cellpainting-gallery/cpg0016-jump/"
#     "{Metadata_Source}/workspace/profiles/"
#     "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
# )

csv_formatter = (
    "{0}/cpg0016-jump/{2}.csv")
    # "{Metadata_Source}/workspace/profiles/"
    # "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.csv"

local_formatter = (
    "{0}/cpg0016-jump/"
    "{1}/workspace/profiles/"
    "{2}/{3}/{3}.parquet"
)
