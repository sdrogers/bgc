import sys
from glob import glob
import os
import sys
import re

from multiprocessing import Pool, cpu_count
from collections import defaultdict

def run_bigscape_hmmscan(input_dir, output_folder, pfam_dir,
                         bigscape_path, biopython_path, parallel=False):

    sys.path.append(bigscape_path)
    sys.path.append(biopython_path)

    import bigscape as bs
    import functions as f

    class bgc_data:
        def __init__(self, accession_id, description, product, records, max_width, organism, taxonomy, biosynthetic_genes,
                     contig_edge):
            # These two properties come from the genbank file:
            self.accession_id = accession_id
            self.description = description
            # AntiSMASH predicted class of compound:
            self.product = product
            # number of records in the genbank file (think of multi-locus BGCs):
            self.records = records
            # length of largest record (it will be used for ArrowerSVG):
            self.max_width = int(max_width)
            # organism
            self.organism = organism
            # taxonomy as a string (of comma-separated values)
            self.taxonomy = taxonomy
            # Internal set of tags corresponding to genes that AntiSMASH marked
            # as "Kind: Biosynthetic". It is formed as
            # clusterName + "_ORF" + cds_number + ":gid:" + gene_id + ":pid:" + protein_id + ":loc:" + gene_start + ":" + gene_end + ":strand:" + {+,-}
            self.biosynthetic_genes = biosynthetic_genes
            # AntiSMASH 4+ marks BGCs that sit on the edge of a contig
            self.contig_edge = contig_edge


    f.create_directory(output_folder, "Output", False)
    bgc_fasta_folder = os.path.join(output_folder, "fasta")
    f.create_directory(bgc_fasta_folder, "BGC fastas", False)

    bs.bgc_data = bgc_data
    bs.mode = 'global'

    bgc_info = {} # Stores, per BGC: predicted type, gbk Description,
                  # number of records, width of longest record,
                  # GenBank's accession, Biosynthetic Genes' ids

    min_bgc_size = 0 # Provide the minimum size of a BGC to be included in the analysis. Default is 0 base pairs
    exclude_gbk_str = '' # If this string occurs in the gbk filename, this file will not be used for the analysis

    # genbankDict: {cluster_name:[genbank_path_to_1st_instance,[sample_1,sample_2,...]]}
    genbankDict = bs.get_gbk_files(input_dir, output_folder, bgc_fasta_folder,
                                min_bgc_size, exclude_gbk_str, bgc_info)

    # clusters and sampleDict contain the necessary structure for all-vs-all and sample analysis
    clusters = genbankDict.keys()
    clusterNames = tuple(sorted(clusters))

    sampleDict = {} # {sampleName:set(bgc1,bgc2,...)}
    gbk_files = [] # raw list of gbk file locations
    for (cluster, (path, clusterSample)) in genbankDict.items():
        gbk_files.append(path)
        for sample in clusterSample:
            clustersInSample = sampleDict.get(sample, set())
            clustersInSample.add(cluster)
            sampleDict[sample] = clustersInSample

    baseNames = set(clusters)

    allFastaFiles = set(glob(os.path.join(bgc_fasta_folder,"*.fasta")))
    fastaFiles = set()
    for name in baseNames:
        fastaFiles.add(os.path.join(bgc_fasta_folder, name+".fasta"))
    fastaBases = allFastaFiles.intersection(fastaFiles)
    task_set = fastaFiles
    verbose = False

    domtable_folder = os.path.join(output_folder, "domtable")
    f.create_directory(domtable_folder, "Domtable", False)

    if parallel:
        cores = cpu_count()
        pool = Pool(cores, maxtasksperchild=1)
        for fasta_file in task_set:
            pool.apply_async(bs.runHmmScan, args=(fasta_file, pfam_dir, domtable_folder, verbose))
        pool.close()
        pool.join()
    else:
        i = 1
        for fasta_file in task_set:
            print 'Processing %d/%d' % (i, len(task_set))
            bs.runHmmScan(fasta_file, pfam_dir, domtable_folder, verbose)
            i += 1

    print("Processing domtable files")

    pfs_folder = os.path.join(output_folder, "pfs")
    pfd_folder = os.path.join(output_folder, "pfd")
    f.create_directory(pfs_folder, "pfs", False)
    f.create_directory(pfd_folder, "pfd", False)

    allDomtableFiles = set(glob(os.path.join(domtable_folder,"*.domtable")))
    domtableFiles = set()
    for name in baseNames:
        domtableFiles.add(os.path.join(domtable_folder, name+".domtable"))
    domtableBases = allDomtableFiles.intersection(domtableFiles)
    alreadyDone = set()

    bs.gbk_files = gbk_files
    bs.genbankDict = genbankDict
    bs.clusters = clusters
    bs.baseNames = baseNames
    bs.sampleDict = sampleDict

    # Specify at which overlap percentage domains are considered to overlap.
    # Domain with the best score is kept (default=0.1).
    domain_overlap_cutoff = 0.1
    for domtableFile in domtableFiles - alreadyDone:
        try:
            bs.parseHmmScan(domtableFile, pfd_folder, pfs_folder, domain_overlap_cutoff)
        except IndexError:
            continue
        except ValueError:
            continue

    return baseNames

def get_matching_line(fn, bigscape_path, biopython_path):

    sys.path.append(bigscape_path)
    sys.path.append(biopython_path)

    import bigscape as bs
    import functions as f

    products = []
    regex = re.compile('product="([A-za-z0-9-\_ ]+)"') # 'product=', followed by any alphanumeric, _, - and space
    with open(fn) as ff:
        for line in ff:
            result = regex.findall(line)
            if len(result) > 0:
                product = result[0].strip()
                products.append(product)

    if len(products) == 1: # assume this is the 'product' in the cluster section
        predicted = f.sort_bgc(product)
        return product, predicted

    elif len(products) == 2: # both 'cluster' and 'CDS' have products
        predicted0 = f.sort_bgc(products[0])
        if len(predicted0) > 0:
            return products[0], predicted0
        else:
            predicted1 = f.sort_bgc(products[1])
            return products[1], predicted1

    elif len(products) > 0: # just take the first one
        predicted0 = f.sort_bgc(products[0])
        return products[0], predicted0

    else:
        return "Others", "Others"


def get_products(input_dir, bigscape_path, biopython_path):

    bigscape_products = {}
    antismash_products = {}
    gbk_filenames = set(glob(os.path.join(input_dir, '*.gbk')))
    for fn in gbk_filenames:
        basename = os.path.splitext(os.path.basename(fn))[0]
        product, predicted = get_matching_line(fn, bigscape_path, biopython_path)
        antismash_products[basename] = product
        bigscape_products[basename] = predicted

    return bigscape_products, antismash_products