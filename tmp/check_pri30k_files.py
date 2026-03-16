#!/usr/bin/env python3
"""
检查 PRI30k split CSV 中每一行对应的 .cif 文件是否存在，并做简单的链存在性与大小检查。
用法:
    python tmp/check_pri30k_files.py --config config/datasets/PRI30k.yml

输出: tmp/pri30k_file_report.csv
"""
import argparse
import os
import csv
import sys
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def find_file_variants(base_dir, pdb_id, prot_chain, na_chain):
    """尝试若干 filename 变体，返回第一个存在的路径及变体说明。"""
    candidates = []
    # 原始构造
    orig = f"{pdb_id}_{prot_chain}_{na_chain}.cif"
    candidates.append((orig, 'orig'))
    # 大小写变体
    candidates.append((f"{pdb_id.upper()}_{prot_chain.upper()}_{na_chain.upper()}.cif", 'upper_all'))
    candidates.append((f"{pdb_id.lower()}_{prot_chain.lower()}_{na_chain.lower()}.cif", 'lower_all'))
    # 仅 pdb id 大写
    candidates.append((f"{pdb_id.upper()}_{prot_chain}_{na_chain}.cif", 'pdb_upper'))
    candidates.append((f"{pdb_id.lower()}_{prot_chain}_{na_chain}.cif", 'pdb_lower'))
    # 仅链大写/小写
    candidates.append((f"{pdb_id}_{prot_chain.upper()}_{na_chain}.cif", 'prot_chain_upper'))
    candidates.append((f"{pdb_id}_{prot_chain.lower()}_{na_chain}.cif", 'prot_chain_lower'))
    candidates.append((f"{pdb_id}_{prot_chain}_{na_chain.upper()}.cif", 'na_chain_upper'))
    candidates.append((f"{pdb_id}_{prot_chain}_{na_chain.lower()}.cif", 'na_chain_lower'))

    for name, note in candidates:
        p = Path(base_dir) / name
        if p.exists():
            return str(p), note
    # 也尝试没有链后缀的文件
    p = Path(base_dir) / f"{pdb_id}.cif"
    if p.exists():
        return str(p), 'pdb_only'
    return None, None


def simple_chain_search(filepath, chain):
    """在文件内搜索链标记的简单计数（对 mmCIF 和 PDB 都能工作作为启发式方法）。"""
    if chain is None or chain == '':
        return 0
    try:
        with open(filepath, 'r', errors='ignore') as f:
            txt = f.read()
    except Exception:
        return -1
    # mmCIF 中链 id 常在 label_asym_id 字段，也可以直接搜索单个字符出现次数（启发式）
    count = txt.count(chain)
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/datasets/PRI30k.yml')
    p.add_argument('--out', default='tmp/pri30k_file_report.csv')
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    df_path = cfg.get('df_path')
    data_root = cfg.get('data_root')
    col_prot = cfg.get('col_prot_name', 'PDB')
    col_prot_chain = cfg.get('col_prot_chain', 'Protein chains')
    col_na_chain = cfg.get('col_na_chain', 'RNA chains')

    df = pd.read_csv(df_path)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    rows = []
    missing = 0
    bad_chain = 0
    total = len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Scanning rows'):
        pdb = str(row[col_prot]).strip()
        # 处理可能的逗号分隔链列表，取第一个链（与数据加载器行为一致）
        prot_chain = str(row[col_prot_chain]).split(',')[0].strip()
        na_chain = str(row[col_na_chain]).split(',')[0].strip()
        expected_name = f"{pdb}_{prot_chain}_{na_chain}.cif"
        path, note = find_file_variants(data_root, pdb, prot_chain, na_chain)
        exists = bool(path)
        size = 0
        prot_cnt = 0
        na_cnt = 0
        file_note = note
        if not exists:
            missing += 1
            file_note = 'missing_file'
        else:
            try:
                size = Path(path).stat().st_size
                prot_cnt = simple_chain_search(path, prot_chain)
                na_cnt = simple_chain_search(path, na_chain)
                # 如果链出现在文件中的次数很少，可能链标记不匹配或文件不含该链
                if (prot_cnt < 5) or (na_cnt < 5):
                    bad_chain += 1
                    if file_note:
                        file_note = file_note + ';low_chain_count'
                    else:
                        file_note = 'low_chain_count'
            except Exception as e:
                file_note = f'open_error:{e}'

        rows.append({
            'index': idx,
            'pdb': pdb,
            'prot_chain': prot_chain,
            'na_chain': na_chain,
            'expected_filename': expected_name,
            'found_path': path if path else '',
            'exists': exists,
            'size': size,
            'prot_chain_count': prot_cnt,
            'na_chain_count': na_cnt,
            'note': file_note,
        })

    # 写出报告
    keys = ['index','pdb','prot_chain','na_chain','expected_filename','found_path','exists','size','prot_chain_count','na_chain_count','note']
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Scanned {total} rows. Missing files: {missing}. Low-chain-counts: {bad_chain}.")
    print(f"Report written to {args.out}")


if __name__ == '__main__':
    main()
