#!/usr/bin/env python3
"""
预缓存 PRI30k 解析数据脚本。
将按 CSV（config 指定的 df_path）逐行调用数据集的解析并把结果写入 diskcache。
输出：在指定 cache_dir 下写入缓存；失败项写入 tmp/pri30k_precache_fail.csv。

用法：
  python -u tmp/precache_pri30k.py --config tmp/pri30k_filtered.yml --cache_dir ./cache/pri30k
"""
import argparse
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import diskcache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/datasets/PRI30k.yml')
    p.add_argument('--cache_dir', default='./cache/pri30k')
    p.add_argument('--out_fail', default='tmp/pri30k_precache_fail.csv')
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    df_path = cfg.get('df_path')
    data_root = cfg.get('data_root')
    # allow override df_path from cfg
    if df_path is None:
        raise RuntimeError('df_path not found in config')

    df = pd.read_csv(df_path)

    cache_dir = Path(args.cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    cache = diskcache.Cache(directory=str(cache_dir), eviction_policy='none')

    # import here to ensure package path
    from data.pri30k_dataset import PRI30kDataset
    dataset = PRI30kDataset(df, data_root=data_root, diskcache=cache, transform=None)

    fail_rows = []
    total = len(df)
    for i in tqdm(range(total), desc='Pre-caching PRI30k'):
        try:
            cplx = dataset.load_data(i)
            if cplx is None:
                row = df.iloc[i]
                fail_rows.append({'index': i, 'pdb': row.get('PDB', ''), 'prot_chain': row.get('Protein chains', ''), 'na_chain': row.get('RNA chains', '' )})
        except Exception as e:
            row = df.iloc[i]
            fail_rows.append({'index': i, 'pdb': row.get('PDB', ''), 'prot_chain': row.get('Protein chains', ''), 'na_chain': row.get('RNA chains', '' ), 'error': str(e)})

    if len(fail_rows) > 0:
        pd.DataFrame(fail_rows).to_csv(args.out_fail, index=False)
        print(f"Precache finished. {len(fail_rows)} failures. See {args.out_fail}")
    else:
        print("Precache finished. No failures.")

    cache.close()

if __name__ == '__main__':
    main()
