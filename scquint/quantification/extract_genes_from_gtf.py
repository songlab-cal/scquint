#!/usr/bin/env python

import sys

for line in sys.stdin:
    try:
        attr = dict(item.strip().split(' ') for item in line.split('\t')[8].strip('\n').split(';')[:10] if item)
    except:
        sys.stderr.write(line)
        raise Exception("debug")
    #sys.stdout.write("%s\n" % attr['gene_name'].strip('\"'))
    sys.stdout.write("%s\n" % attr['gene_id'].strip('\"'))
