#Purpose: This script generates jobs.py files and creates a somaWF file 
#containing the jobs to be send to the cluster with soma_workflow.
#Project: MenRot
#Author: Darinka Truebutschek
#Date: 08 Nov 2016

########################################################################
#Import necessary libraries
import os

from soma_workflow.client import Job, Workflow, Helper

cwd = os.path.dirname(os.path.abspath(__file__)) #only if called from within script, else just give the actual path
#cwd = '/neurospin/meg/meg_tmp/MenRot_Truebutschek_2016/Python_Scripts/Decoding'
os.chdir(cwd)

from menRot_newPipeline_loc_cfg import (wkdir)

########################################################################
#List of parameters to be parallelized

ListSubject = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477',
	'bo160176', 'at140305', 'df150397', 'rl130571', 'mm140137', 
	'mb160304', 'lk160274', 'av160302', 'cc150418', 'cs150204', 
	'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 'cc130066',
	'in110286', 'ss120102']

#ListCondition = [['Train_loc', 'Test_AllSeen'], ['Train_loc', 'Test_AllUnseen'],
	#['Train_AllSeen', 'Test_AllSeen'], ['Train_AllUnseen', 'Test_AllUnseen'], 
	#['Train_loc', 'Test_AllSeen'], ['Train_loc', 'Test_AllUnseen'],
	#['Train_AllSeen', 'Test_AllSeen'], ['Train_AllUnseen', 'Test_AllUnseen'],
	#['Train_loc', 'Test_AllSeen'], ['Train_loc', 'Test_AllUnseen'],
	#['Train_AllSeen', 'Test_AllSeen'], ['Train_AllUnseen', 'Test_AllUnseen']]
	
#ListCondition = [['Train_loc', 'Test_AllSeen'], ['Train_loc', 'Test_AllUnseen'],
	#['Train_AllSeen', 'Test_AllSeen'], ['Train_AllUnseen', 'Test_AllUnseen']]

#ListCondition = [['Train_AllSeen', 'Test_AllUnseen'], ['Train_AllUnseen', 'Test_AllSeen']]

ListCondition = [['Train_All', 'Test_All']]

#ListDecCond = ['loc', 'loc', 'loc', 'loc', 
	#'resp', 'resp', 'resp', 'resp',
	#'infer', 'infer', 'infer', 'infer']

ListDecCond = ['loc', 'infer', 'resp']
#ListDecCond = ['locInfer', 'locResp', 'inferLoc', 'inferResp', 'respLoc', 'respInfer']

########################################################################
#Initialize job files and names
List_python_files = []

initbody = 'import sys \n'
initbody = initbody + "sys.path.append(" + "'" + cwd + "')\n"
initbody = initbody + 'import menRot_newPipeline_subscore_decLoc_EOG as DecInd\n'

#Write actual job files
python_file, Listfile, ListJobName = [], [], []

for d, decCond in enumerate(ListDecCond):
	for c, condcouple in enumerate(ListCondition):
		for s, subject in enumerate(ListSubject):
			
			tmp1 = condcouple[0].find('_')
			tmp2 = condcouple[1].find('_')
			string1 = condcouple[0][0 : tmp1] + condcouple[0][tmp1 + 1 :]
			string2 = condcouple[1][0 : tmp2] + condcouple[1][tmp2 + 1 :]	
			my_filename = decCond[0].upper() + decCond[1 :] + '_' + string1 + '_' + string2

			body = initbody + "DecInd.menRot_newPipeline_subscore_decLoc_EOG('" + wkdir + "',"
			body = body + str(condcouple) + ","
			body = body + "'" + subject + "'," 
			body = body + "'" + my_filename + "',"
			body = body + "'" + str(decCond) + "')"
		
			#Use a transparent and complete job name referring to arguments of interests
			jobname = subject
		
			for cond in condcouple:
				jobname = jobname + '_' + cond + '_' + decCond
			ListJobName.append(jobname)

			#Write jobs in a dedicated folder
			name_file = []
			name_file = os.path.join(wkdir, ('SomaWF/Decoding/Jobs/jobs_' + jobname + '.py'))
			Listfile.append(name_file)
		
			with open(name_file, 'w') as python_file:
				python_file.write(body)

########################################################################
#Create workflow
jobs = []

for i in range(len(Listfile)):
	JobVar = Job(command = ['qsub -- /home/dt237143/anaconda2/bin/python '+Listfile[i]], name = ListJobName[i],
                 native_specification = '-l walltime=12:00:00, -l nodes=1:ppn=2')
	jobs.append(JobVar)

#Save the workflow variables
WfVar = Workflow(jobs = jobs)
somaWF_name = os.path.join(wkdir, 'SomaWF/Decoding/menRot_decLoc_workflow_EOG')
Helper.serialize(somaWF_name, WfVar)
