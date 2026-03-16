#!/usr/bin/env python3
"""
根据 tmp/pri30k_file_report.csv 生成一个过滤后的 CSV（仅保留存在且链计数足够的项）。
输出: tmp/pri30k_filtered.csv, tmp/pri30k_filtered_report.csv
"""
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', default='tmp/pri30k_file_report.csv')
    p.add_argument('--out', default='tmp/pri30k_filtered.csv')
    p.add_argument('--detail', default='tmp/pri30k_filtered_report.csv')
    p.add_argument('--min_chain_count', type=int, default=5)
    args = p.parse_args()

    df = pd.read_csv(args.report)
    keep_rows = []
    dropped = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='Filtering rows'):
        exists = bool(r['exists'])
        prot_cnt = int(r['prot_chain_count']) if not pd.isna(r['prot_chain_count']) else 0
        na_cnt = int(r['na_chain_count']) if not pd.isna(r['na_chain_count']) else 0
        note = str(r.get('note', ''))
        if exists and prot_cnt >= args.min_chain_count and na_cnt >= args.min_chain_count:
            keep_rows.append(r['index'])
        else:
            dropped.append({
                'index': r['index'],
                'pdb': r['pdb'],
                'prot_chain': r['prot_chain'],
                'na_chain': r['na_chain'],
                'note': note,
                'prot_chain_count': prot_cnt,
                'na_chain_count': na_cnt,
            })

    # 读取原始 CSV（默认从 config）——尝试从 config path not available here, so assume original in config path
    # 简单地读取 datasets/PRI30k/splits/pretrain_length_750_clean.csv
    orig = Path('datasets/PRI30k/splits/pretrain_length_750_clean.csv')
    if not orig.exists():
        print(f"Original CSV not found at {orig}. Can't write filtered dataset CSV.")
        # 但仍写出 detail report
        pd.DataFrame(dropped).to_csv(args.detail, index=False)
        return

    df_orig = pd.read_csv(orig)
    # 保持行顺序，按 index 字段过滤
    df_filtered = df_orig.loc[keep_rows]
    df_filtered.to_csv(args.out, index=False)
    pd.DataFrame(dropped).to_csv(args.detail, index=False)
    print(f"Filtered CSV written to {args.out}. Kept {len(keep_rows)} rows, dropped {len(dropped)} rows. Detail: {args.detail}")

if __name__ == '__main__':
    main()
