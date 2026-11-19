#!/usr/bin/env python3
import gzip,json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_instances(path):
    data=json.load(gzip.open(path,'rt'))
    rows=[]
    for inst in data.get('instances',[]):
        s=inst.get('precomputed_behavior',{}).get('summary',{})
        t=s.get('trigger') or {}
        rows.append({
            'scenario':path.stem,'instance_id':inst.get('id'),'window_id':inst.get('window_id'),
            'focus_track_id':inst.get('focus_track_id'),'path_label':inst.get('robot_path_label'),
            'result':s.get('result'),'outcome':s.get('outcome'),'false_positive':s.get('false_positive'),
            'final_action':s.get('final_action'),'collision':s.get('collision'),'near_miss':s.get('near_miss'),
            'failed_test':s.get('failed_test'),'max_risk':s.get('max_risk'),'min_trust':s.get('min_trust'),'min_ttc':s.get('min_ttc'),
            'trigger_action':t.get('action'),'trigger_state':t.get('state'),'trigger_scenario':t.get('scenario'),
            'trigger_frame':t.get('frame'),'trigger_ttc':t.get('ttc'),'trigger_trust':t.get('trust'),
            'route_prob':t.get('route_prob'),'route_uncertainty':t.get('route_uncertainty'),
            'maneuver':t.get('maneuver'),'maneuver_confidence':t.get('maneuver_confidence'),
            'supervisors_enabled':s.get('supervisors_enabled',True)
        })
    return rows

def bar(series,title,out):
    plt.figure(figsize=(8,4)); series.plot(kind='bar'); plt.title(title); plt.tight_layout(); plt.savefig(out); plt.close()

def plot(df,out):
    bar(df.result.value_counts().sort_index(),'PASS vs FAIL',out/'result_distribution.png')
    if 'outcome' in df and df.outcome.notna().any():
        bar(df.outcome.fillna('UNKNOWN').value_counts(),'Outcome Breakdown',out/'outcome_distribution.png')

    d=df.dropna(subset=['trigger_ttc'])
    if len(d):
        plt.figure(figsize=(8,4)); plt.hist(d.trigger_ttc,bins=25)
        plt.axvline(2.0,color='r',ls='--',label='Late threshold'); plt.legend()
        plt.xlabel('Trigger TTC [s]'); plt.tight_layout(); plt.savefig(out/'trigger_ttc_distribution.png'); plt.close()

    d=df.dropna(subset=['trigger_ttc','trigger_trust'])
    if len(d):
        plt.figure(figsize=(6,5)); plt.scatter(d.trigger_trust,d.trigger_ttc,alpha=.6)
        plt.xlabel('Trigger Trust'); plt.ylabel('Trigger TTC [s]'); plt.tight_layout(); plt.savefig(out/'trust_vs_ttc.png'); plt.close()

    d=df[df.trigger_scenario.notna() & (df.trigger_scenario!='')]
    if len(d):
        bar(d.trigger_scenario.value_counts(),'Scenario Types',out/'scenario_distribution.png')

    d=df.groupby('path_label')['trigger_ttc'].median().dropna().sort_values()
    if len(d):
        plt.figure(figsize=(10,max(4,len(d)*0.22))); d.plot(kind='barh')
        plt.xlabel('Median Trigger TTC [s]'); plt.tight_layout(); plt.savefig(out/'path_comparison.png'); plt.close()

    if df.supervisors_enabled.nunique()>1:
        c=df.groupby(['supervisors_enabled','result']).size().unstack(fill_value=0)
        c.plot(kind='bar',figsize=(8,4)); plt.tight_layout(); plt.savefig(out/'pass_fail_comparison.png'); plt.close()


def main():
    files=sorted(Path('./scenario_data').glob('scenario_*_instances.json.gz')) or sorted(Path('.').glob('scenario_*_instances.json.gz'))
    if not files: raise FileNotFoundError('No scenario_*_instances.json.gz files found')
    rows=[]
    for f in files: rows.extend(load_instances(f))
    df=pd.DataFrame(rows)
    out=files[0].parent/'scenario_analysis'; out.mkdir(exist_ok=True)
    df.to_csv(out/'scenario_summary.csv',index=False)
    plot(df,out)
    print('RESULTS',df.result.value_counts())
    if 'outcome' in df: print('OUTCOMES',df.outcome.fillna('UNKNOWN').value_counts())
    print(f'Analysis written to {out}')

if __name__=='__main__': main()
