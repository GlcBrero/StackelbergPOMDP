"""Draw the platform policy grid and current-policy learning curve."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib.patches import Patch

import numpy as np

from .style import DEEP_PALETTE, save_paper_figure
from .data import csv_rows, platform_learning

LEARNED_BLUE=DEEP_PALETTE[0]

COMPARATOR_ORANGE=DEEP_PALETTE[1]

def payoffs(prices,rule,i,j,cost):
    cap=int(rule[i,j])
    w=np.asarray([math.exp((2-prices[k])/.25) if k<=cap else 0. for k in (i,j)])
    demand=w/(1+w.sum())
    return (prices[[i,j]]-cost)*demand, .25*math.log(1+w.sum())

def draw_policy(ax, prices, rule, outcomes):
    """Keep every grid cell while labeling only a few price coordinates."""
    m=len(prices)
    first,second=np.meshgrid(np.arange(m),np.arange(m),indexing='ij')
    admitted=(first<=rule).astype(int)+(second<=rule).astype(int)
    assert np.all((admitted!=1)|(first!=second))
    colors=['#F3F3F3','#A9C2DA',LEARNED_BLUE]
    edges=np.arange(-.5,m+.5,1)
    ax.pcolormesh(edges,edges,admitted.T,cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(-.5,3.5),3),shading='flat',
                  edgecolors='#d0d0d0',linewidth=.25)
    ax.set_aspect('equal')
    ticks=[0,4,7,11,14]
    labels=[f'{prices[k]:.3f}' for k in ticks]
    ax.set_xticks(ticks);ax.set_yticks(ticks)
    ax.set_xticklabels(labels,fontsize=8.5);ax.set_yticklabels(labels,fontsize=8.5)
    ax.set_xticks(np.arange(-.5,m,1),minor=True);ax.set_yticks(np.arange(-.5,m,1),minor=True)
    ax.grid(False);ax.grid(which='minor',color='white',linewidth=.45)
    ax.tick_params(which='minor',bottom=False,left=False)
    ax.set_xlabel('Seller 1 price',fontsize=10);ax.set_ylabel('Seller 2 price',fontsize=10)
    ax.plot([-.5,m-.5],[-.5,m-.5],color='#777777',linestyle=':',linewidth=.6)
    for cost,label in [('1.0','L'),('1.5','H')]:
        first,second=outcomes[cost]['price_indices']
        ax.scatter(first,second,s=78,facecolor='white',edgecolor='black',linewidth=.9,zorder=5)
        ax.text(first,second,label,ha='center',va='center',weight='bold',fontsize=8,zorder=6)
    ax.legend(handles=[Patch(facecolor=colors[k],edgecolor='#777777',linewidth=.4,label=l)
                       for k,l in [(2,'Both'),(1,'Lower price only'),(0,'Neither')]],
              loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,fontsize=8.5,
              handlelength=1.,columnspacing=.9)
    ax.set_title('(a) Learned policy',fontsize=11,pad=10)

def draw_learning(ax, x, curves, results, baselines, uniform):
    fixed_cap=results['validation_fixed_cap_cs']
    average=curves.mean(0);se=curves.std(0,ddof=1)/np.sqrt(len(curves))
    for values in curves:
        ax.plot(x,values,linewidth=.75,alpha=.25,color=LEARNED_BLUE)
    ax.plot(x,average,color=LEARNED_BLUE,linewidth=2.0,
            label='PPO',zorder=5)
    ax.fill_between(x,average-se,average+se,color=LEARNED_BLUE,alpha=.17,linewidth=0)
    lines=[('Fixed cap',fixed_cap,'#4A4A4A',':'),
           ('Clipped rule',results['validation_clipped_cs'],COMPARATOR_ORANGE,'--'),
           ('Random search: selected policy',uniform['validation_cs'],'#6B6B6B',(0,(5,2,1,2))),
           ('No intervention',baselines['no_intervention']['mean'],'#909090',(0,(1,2)))]
    for label,value,color,style in lines:
        ax.axhline(value,color=color,linestyle=style,linewidth=1.35,label=label)
    lower=max(0.,min(float(curves.min())-.03,
              float((average-se).min())-.03,
              min(value for _,value,_,_ in lines)-.02))
    upper=max(float(curves.max())+.03,
              float((average+se).max())+.02,
              max(value for _,value,_,_ in lines)+.03)
    ax.set(xlim=(0,20000),ylim=(lower,upper))
    ax.set_xlabel('Seller-response evaluations',fontsize=10)
    ax.set_ylabel('Validation consumer surplus',fontsize=10)
    ax.set_xticks([0,5000,10000,15000,20000])
    ax.set_xticklabels(['0','5,000','10,000','15,000','20,000'],fontsize=8.5)
    ax.tick_params(axis='y',labelsize=9)
    ax.legend(loc='upper center',bbox_to_anchor=(.46,-.19),ncol=2,fontsize=8.0,
              handlelength=1.4,columnspacing=.7,labelspacing=.3)
    ax.set_title('(b) PPO training',fontsize=11,pad=10)

def save_panel(fig, output):
    # Fixed canvas sizes align the two independently reproducible LaTeX panels.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        xticks=[label for position,label in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0]<=position<=ax.get_xlim()[1]]
        yticks=[label for position,label in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=position<=ax.get_ylim()[1]]
        labels=[ax.title,ax.xaxis.label,ax.yaxis.label,*xticks,*yticks,ax.get_legend()]
        for label in labels:
            if label is None or not label.get_visible():
                continue
            box=label.get_window_extent(renderer)
            assert box.x0>=-.5 and box.y0>=-.5 and box.x1<=fig.bbox.x1+.5 and box.y1<=fig.bbox.y1+.5, (output,box.bounds)
    with plt.rc_context({'savefig.bbox':None}):
        save_paper_figure(fig,output)
    plt.close(fig)


def load_search():
    records = [r for r in csv_rows('table_platform_results') if r['method'] == 'Random search']
    seeds = sorted({r['seller_seed'] for r in records})
    cells = {(r['cost'], r['seller_seed']): r['consumer_surplus'] for r in records}
    assert len(records) == len(cells) == 50 and len(seeds) == 25
    assert set(cells) == {(c, s) for c in (1., 1.5) for s in seeds}
    values = np.asarray([[cells[c, s] for c in (1., 1.5)] for s in seeds])
    for field in ('search_candidate', 'search_selection_score', 'search_final_sha256'):
        assert len({r[field] for r in records}) == 1
    selected = records[0]
    mean = float(values.mean())
    assert abs(mean-selected['policy_mean_cs']) < 1e-12
    return dict(cs=mean, per_cost={str(c): float(values[:, i].mean()) for i, c in enumerate((1., 1.5))},
        validation_cs=platform_learning()[1]['Random search'],
        fresh_seller_seed_se=float(values.mean(axis=1).std(ddof=1)/np.sqrt(len(values))),
        per_cost_se={str(c): float(values[:, i].std(ddof=1)/np.sqrt(len(values))) for i, c in enumerate((1., 1.5))},
        candidate=selected['search_candidate'], selection_score=selected['search_selection_score'],
        distinct_policies=10000, training_responses=20000, fresh_responses=50,
        plot_validation_responses=10, final_sha256=selected['search_final_sha256'],
        method='uniform_valid_admission_rules_single_pass')
