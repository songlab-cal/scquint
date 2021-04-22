import os
import sys

module_path = os.path.dirname(os.path.realpath(__file__))
snakefile_path = os.path.join(module_path, sys.argv[1])

command = "snakemake --snakefile {} {}".format(snakefile_path, " ".join(sys.argv[2:]))
print(command)
os.system(command)
