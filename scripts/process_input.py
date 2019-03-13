import argparse

parser = argparse.ArgumentParser(description='Gene filter')
parser.add_argument(
    '-i',
    action='append',
    dest='input_path',
    default=[],
    help='Input')
parser.add_argument(
    '-o',
    action='append',
    dest='output_path',
    default=[],
    help='Output')
parser.add_argument(
    '--no_head',
    action='store_true',
    dest='no_head',
    default=False,
    help='If to include head line')

args = parser.parse_args()
input_path = args.input_path[0]
output_path = args.output_path[0]
no_head = args.no_head

with open('/mnt/osf1/user/wuzhq/meta/utils/selected_genes.csv', 'r') as f:
  selected_genes = set([g[:-1] for g in f.readlines()])

f1 = open(input_path, 'r')
f2 = open(output_path, 'w')

for i, line in enumerate(f1):
  if i==0 and not no_head:
    f2.write(line)
  if i>0:
    if line.split()[0] in selected_genes:
      f2.write(line)

f1.close()
f2.close()
