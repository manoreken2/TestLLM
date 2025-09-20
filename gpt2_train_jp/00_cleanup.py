import ntpath
import glob
import re

text_files = glob.glob('../../../texts/*/*.txt')
print(text_files)

def remove_header(s):
    bar = "-------------------------------------------------------"
    idx0 = s.find(bar)
    idx1 = s.find(bar, idx0+1)
    s2 = s[:idx0] + s[idx1+len(bar):]
    return s2

def remove_footer(s):
    idx0 = s.find('底本：')
    s2 = s[:idx0]
    return s2

def remove_meta(s):
    s = re.sub(r"［[^］]*?］", "", s)
    s = re.sub(r"《[^》]*?》", "", s)
    s = re.sub(r"｜", "", s)
    s = re.sub(r"　", " ", s)
    s = re.sub(r"\r\n", "\n", s)

    s = re.sub('――+', '―', s) # 2つ以上連続する―を1個にする。
    s = re.sub('\n\n+', '\n', s)
    s = re.sub('  +', ' ', s)

    from_list = '０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
    to_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    for i in range(len(from_list[0])):
        s = re.sub(from_list[i], to_list[i], s)
    return s

def write(s, read_path):
    fn = ntpath.basename(read_path)
    write_path = "orig/" + fn
    with open(write_path, mode='w',encoding='utf-8') as f:
        f.write(s)

for path in text_files:
    with open(path, 'rb') as fp:
        s = fp.read().decode('shift-jis')
        s = remove_header(s)
        s = remove_footer(s)
        s = remove_meta(s)
        print(s)
        write(s, path)

