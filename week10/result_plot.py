# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import os
import json
import numpy as np
import codecs

RESULT_FOLDER = 'result'
RESULT_SIM = 'result_sim'
RESULT_DIS = 'result_dis'
RESULT_TRU = 'result_tru'

def box_plot():
    result_sim = os.listdir(RESULT_SIM)
    result_dis = os.listdir(RESULT_DIS)
    result_tru = os.listdir(RESULT_TRU)

    diss_act_list1,sim_act_list1,true_act_list1,diss_st_list1,sim_st_list1,true_st_list1 = rate_list(result_sim,RESULT_SIM)
    diss_act_list2,sim_act_list2,true_act_list2,diss_st_list2,sim_st_list2,true_st_list2 = rate_list(result_dis,RESULT_DIS)
    diss_act_list3,sim_act_list3,true_act_list3,diss_st_list3,sim_st_list3,true_st_list3 = rate_list(result_tru,RESULT_TRU)

    fig4 = plt.figure(figsize=(8,11))
    ax1 = plt.subplot(3,2,1)
    ax1.set_title("dissimulation action rate average")
    bplot = plt.boxplot([diss_act_list1,diss_act_list2,diss_act_list3],patch_artist=True)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')

    ax2 = plt.subplot(3,2,2)
    ax2.set_title("simulation action rate average")
    bplot2 = plt.boxplot([sim_act_list1,sim_act_list2,sim_act_list3],patch_artist=True)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')

    ax3 = plt.subplot(3,2,3)
    ax3.set_title("truthful action rate average")
    bplot3 = plt.boxplot([true_act_list1,true_act_list2,true_act_list3],patch_artist=True)
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')

    ax4 = plt.subplot(3,2,4)
    ax4.set_title("dissimulation state rate average")
    bplot4 = plt.boxplot([diss_st_list1,diss_st_list2,diss_st_list3],patch_artist=True)
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')

    ax5 = plt.subplot(3,2,5)
    ax5.set_title("simulation state rate average")
    bplot5 = plt.boxplot([sim_st_list1,sim_st_list2,sim_st_list3],patch_artist=True)
    ax5.set_xticks([1, 2, 3])
    ax5.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')

    ax6 = plt.subplot(3,2,6)
    ax6.set_title("truthful state rate average")
    bplot6 = plt.boxplot([true_st_list1,true_st_list2,true_st_list3],patch_artist=True)
    ax6.set_xticks([1, 2, 3])
    ax6.set_xticklabels(['simmulation', 'dissimulation', 'baseline'])
    colors = ['pink', 'lightblue', 'lightgreen']
    for patch, color in zip(bplot6['boxes'], colors):
        patch.set_facecolor(color)
    plt.grid(axis='y')
    plt.show()



def rate_list(result,folder):
    ##rate##
    diss_act_list = []
    sim_act_list = []
    true_act_list = []
    diss_st_list = []
    sim_st_list = []
    true_st_list = []
    for result_file in result:
        path = os.path.join(folder,result_file)
        with codecs.open(path,'r') as f:
            result_dict = json.load(f)
            count_actions = len(result_dict['action list'])
            diss_act_list.append(round(int(result_dict['Dissimulation action amount'])/count_actions,3))
            sim_act_list.append(round(int(result_dict['Simulation action amount'])/count_actions,3))
            true_act_list.append(round(int(result_dict['Truthful action amount'])/count_actions,3))
            
            count_states = len(result_dict['state list'])
            diss_st_list.append(round(int(result_dict['Dissimulation state amount'])/count_states,3))
            sim_st_list.append(round(int(result_dict['Simulation state amount'])/count_states,3))
            true_st_list.append(round(int(result_dict['Truthful state amount'])/count_states,3))
    
    return (diss_act_list,sim_act_list,true_act_list,diss_st_list,sim_st_list,true_st_list)


def bar_plot():
    name_list = ['Simulation algorithm','Dissimulation algorithm','Baseline algorithm']
    #sim_path = os.path.join(RESULT_FOLDER,RESULT_SIM)
    #dis_path = os.path.join(RESULT_FOLDER,RESULT_DIS)
    #tru_path = os.path.join(RESULT_FOLDER,RESULT_TRU)
#    result_sim = os.listdir(RESULT_SIM)
#    result_dis = os.listdir(RESULT_DIS)
#    result_tru = os.listdir(RESULT_TRU)

    diss_act_avg1,sim_act_avg1,true_act_avg1,diss_st_avg1,sim_st_avg1,true_st_avg1 = rate_summary(RESULT_SIM)
    diss_act_avg2,sim_act_avg2,true_act_avg2,diss_st_avg2,sim_st_avg2,true_st_avg2 = rate_summary(RESULT_DIS)
    diss_act_avg3,sim_act_avg3,true_act_avg3,diss_st_avg3,sim_st_avg3,true_st_avg3 = rate_summary(RESULT_TRU)

    diss_act_avg = [diss_act_avg1,diss_act_avg2,diss_act_avg3]
    sim_act_avg = [sim_act_avg1,sim_act_avg2,sim_act_avg3]
    true_act_avg =[true_act_avg1,true_act_avg2,true_act_avg3]
    diss_st_avg =[diss_st_avg1,diss_st_avg2,diss_st_avg3]
    sim_st_avg = [sim_st_avg1,sim_st_avg2,sim_st_avg3]
    true_st_avg =[true_st_avg1,true_st_avg2,true_st_avg3]
    
    x = list(range(len(name_list)))
    fig = plt.figure(figsize=(10,10))
    ax1 = plt.subplot(2,2,1)
    name_list = ['simmulation','dissimulation','baseline']
    ax1.set_title("action rate compare")
    plt.bar(x,diss_act_avg,label='dissimulation action rate average',color='lightskyblue')
    plt.bar(x,sim_act_avg,bottom=diss_act_avg,label='simulation action rate average',color='lightcoral')
    plt.bar(x,true_act_avg,bottom=sim_act_avg,label='truthful action rate average',color='orange',tick_label=name_list)
    plt.legend()
    
    ax2 = plt.subplot(2,2,2)
    name_list = ['simmulation','dissimulation','baseline']
    ax2.set_title("state rate compare")
    plt.bar(x,diss_st_avg,label='dissimulation state rate average',color='mediumaquamarine')
    plt.bar(x,sim_st_avg,bottom=diss_st_avg,label='simulation state rate average',color='dodgerblue')
    plt.bar(x,true_st_avg,bottom=sim_st_avg,label='truthful state rate average',tick_label=name_list,color='crimson')
    plt.legend()
    
    ax3 = plt.subplot(2,2,3)
    name_list = ['simmulation','dissimulation','baseline']
    ax3.set_title("compare summary for action rate")
    x = list(range(len(diss_act_avg)))
    total_width,n = 0.9,3
    width = total_width / n
    plt.bar(x,diss_act_avg,width = width, label='dissimulation action rate average',color='lightskyblue')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x,sim_act_avg,width = width,label='simulation action rate average',color='lightcoral')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x,true_act_avg,width = width,label='truthful action rate average',tick_label=name_list,color='orange')
    plt.legend()

    ax4 = plt.subplot(2,2,4)
    name_list = ['simmulation','dissimulation','baseline']
    ax4.set_title("compare summary for state rate")
    total_width1,n1 = 0.9,3
    width1 = total_width1 / n1
    x1 = list(range(len(diss_st_avg)))
    plt.bar(x1,diss_st_avg,width = width, label='dissimulation state rate average',color='mediumaquamarine')
    for i in range(len(x)):
        x1[i] = x1[i] + width1
    plt.bar(x1,sim_st_avg,width = width,label='simulation state rate average',color='dodgerblue')
    for i in range(len(x)):
        x1[i] = x1[i] + width1
    plt.bar(x1,true_st_avg,width = width,label='truthful state rate average',tick_label=name_list,color='crimson')
    plt.legend()
    plt.show()



def rate_summary(folder):
    result = os.listdir(folder)
    exp_times = len(result)
    ##rate##
    diss_act_sum = 0
    sim_act_sum = 0
    true_act_sum = 0
    diss_st_sum = 0
    sim_st_sum = 0
    true_st_sum = 0
    for result_file in result:
        path = os.path.join(folder,result_file)
        with codecs.open(path,'rU') as f:
            result_dict = json.load(f)
            count_actions = len(result_dict['action list'])
            diss_act_sum += round(int(result_dict['Dissimulation action amount'])/count_actions,3)
            sim_act_sum += round(int(result_dict['Simulation action amount'])/count_actions,3)
            true_act_sum += round(int(result_dict['Truthful action amount'])/count_actions,3)
            
            count_states = len(result_dict['state list'])
            diss_st_sum += round(int(result_dict['Dissimulation state amount'])/count_states,3)
            sim_st_sum += round(int(result_dict['Simulation state amount'])/count_states,3)
            true_st_sum += round(int(result_dict['Truthful state amount'])/count_states,3)

    diss_act_avg = diss_act_sum/exp_times
    sim_act_avg = sim_act_sum/exp_times
    true_act_avg = true_act_sum/exp_times
    diss_st_avg = diss_st_sum/exp_times
    sim_st_avg = sim_st_sum/exp_times
    true_st_avg = true_st_sum/exp_times

    return (diss_act_avg,sim_act_avg,true_act_avg,diss_st_avg,sim_st_avg,true_st_avg)



def cdf_plot():
    not_true_act_pro = list()
    not_true_st_pro = list()
    
    result = os.listdir(RESULT_FOLDER)
    for result_file in result:
        with open('./result/'+result_file,'r') as f:
            result_dict = json.load(f)
            first_true_act = result_dict['first truthful act']
            first_true_st = result_dict['first truthful state']
            action_list = result_dict['action list']
            state_list = result_dict['state list']
            count_act = 0
            count_st = 0
            for i in range(len(action_list)):
                if first_true_act == action_list[i]:
                    count_act = i

            for i in range(len(state_list)):
                if first_true_st == state_list[i]:
                    count_st = i

            not_true_act_pro.append((count_act + 1)/len(action_list))
            not_true_st_pro.append((count_st + 1)/len(state_list))
    
    fig2 = plt.figure(figsize=(6,8))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title("CDF for first truthful action")
    n, bins, patches = plt.hist(not_true_act_pro, 100, density=True, histtype='step',
                                           cumulative=True, label='Empirical')
    plt.legend(bbox_to_anchor=(0.65, 0.3), loc=2, borderaxespad=0.)
    
    ax2 = plt.subplot(2,1,2)
    ax2.set_title("CDF for first truthful state")
    n2, bins2, patches2 = plt.hist(not_true_st_pro, 100, density=True, histtype='step',
                                cumulative=True, label='Empirical')
    plt.legend(bbox_to_anchor=(0.65, 0.3), loc=2, borderaxespad=0.)
    plt.show()


def pie_plot():
    result = os.listdir(RESULT_FOLDER)
    exp_times = len(result)
    diss_act_sum = 0
    sim_act_sum = 0
    true_act_sum = 0
    diss_st_sum = 0
    sim_st_sum = 0
    true_st_sum = 0
    for result_file in result:
        path = os.path.join(RESULT_FOLDER,result_file)
        with open(path,'r') as f:
            result_dict = json.load(f)
            count_actions = len(result_dict['action list'])
            diss_act_sum += round(int(result_dict['Dissimulation action amount'])/count_actions,3)
            sim_act_sum += round(int(result_dict['Simulation action amount'])/count_actions,3)
            true_act_sum += round(int(result_dict['Truthful action amount'])/count_actions,3)
            
            count_states = len(result_dict['state list'])
            diss_st_sum += round(int(result_dict['Dissimulation state amount'])/count_states,3)
            sim_st_sum += round(int(result_dict['Simulation state amount'])/count_states,3)
            true_st_sum += round(int(result_dict['Truthful state amount'])/count_states,3)


    act_sum = diss_act_sum + sim_act_sum + true_act_sum
    st_sum = diss_st_sum + sim_st_sum + true_st_sum
    act_avg = [diss_act_sum/act_sum,sim_act_sum/act_sum,true_act_sum/act_sum]
    st_avg = [diss_st_sum/st_sum,sim_st_sum/st_sum,true_st_sum/st_sum]


    fig1 = plt.figure(figsize=(6,6))
    ax1 = plt.subplot(2,1,1)
    ax1.set_title("action rate")
    labels1 = [u'dissimulation action',u'simulation action',u'truthful action']
    sizes1 = [act_avg[0],act_avg[1],act_avg[2]]
    colors1 = ['tomato','lightskyblue','gold']
    explode1 = (0,0,0)
    patches1,text11,text21 = plt.pie(sizes1,explode=explode1,labels=labels1,colors=colors1,autopct = '%3.2f%%',shadow = False,startangle =0,pctdistance = 0.6)


    ax2 = plt.subplot(2,1,2)
    ax2.set_title("state rate")
    labels2 = [u'dissimulation state',u'simulation state',u'truthful state']
    sizes2 = [st_avg[0],st_avg[1],st_avg[2]]
    colors2 = ['tomato','lightskyblue','gold']
    explode2 = (0,0,0)
    patches2,text12,text22 = plt.pie(sizes2,explode=explode2,labels=labels2,colors=colors2,autopct = '%3.2f%%',shadow = False,startangle =0,pctdistance = 0.6)
    plt.show()

def main():
    pie_plot()
    cdf_plot()

    box_plot()
    bar_plot()

if __name__ == '__main__':
    main()
