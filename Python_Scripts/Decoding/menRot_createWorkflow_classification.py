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

from menRot_cfg import (wkdir)

########################################################################
#List of parameters to be parallelized
ListSubject = ['lm130479', 'am150105', 'cb140229', 'nb140272', 'mj100109', 'dp150209', 
	'ag150338', 'ml140071', 'rm080030', 'bl160191', 'lj150477',
	'bo160176', 'at140305', 'df150397', 'rl130571', 'mm140137', 
	'mb160304', 'lk160274', 'av160302', 'cc150418', 'cs150204', 
	'mp110340', 'lg160230', 'mp150285', 'ef160362', 'ml160216', 'pb160320', 'cc130066',
	'in110286', 'ss120102']

ListCondition = [['Train_AllUnseenAcc', 'Test_AllUnseenAcc']]
#ListCondition = [['Train_AllSeen', 'Test_AllUnseenCorr'], ['Train_AllSeen', 'Test_AllUnseenIncorr']]
#ListCondition = [['Train_All', 'Test_All'], ['Train_Rot', 'Test_Rot'], ['Train_noRot', 'Test_noRot'], ['Train_Rot', 'Test_noRot'], ['Train_noRot', 'Test_Rot']]
#ListCondition = [['Train_SeenAbsent', 'Test_SeenAbsent'], ['Train_SeenAbsent', 'Test_UnseenCorrAbsent'], ['Train_SeenAbsent', 'Test_UnseenIncorrAbsent'],
	#['Train_UnseenCorrAbsent', 'Test_SeenAbsent'], ['Train_UnseenCorrAbsent', 'Test_UnseenCorrAbsent'], ['Train_UnseenCorrAbsent', 'Test_UnseenIncorrAbsent'],
	#['Train_UnseenIncorrAbsent', 'Test_SeenAbsent'], ['Train_UnseenIncorrAbsent', 'Test_UnseenCorrAbsent'], ['Train_UnseenIncorrAbsent', 'Test_UnseenIncorrAbsent']]

########################################################################
#Initialize job files and names
List_python_files = []

initbody = 'import sys \n'
initbody = initbody + "sys.path.append(" + "'" + cwd + "')\n"
initbody = initbody + 'import menRot_decVis as DecInd\n'

#Write actual job files
python_file, Listfile, ListJobName = [], [], []

for c, condcouple in enumerate(ListCondition):
	for s, subject in enumerate(ListSubject):
		
		body = initbody + "DecInd.menRot_decVis('" + wkdir + "',"
		body = body + str(condcouple) + ","
		body = body + "'" + subject + "')"
		
		#Use a transparent and complete job name referring to arguments of interests
		jobname = subject
		
		for cond in condcouple:
			jobname = jobname + '_' + cond
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
	JobVar = Job(command = ['python', Listfile[i]], name = ListJobName[i],
                 native_specification = '-l walltime=12:00:00, -l nodes=1:ppn=4')
	jobs.append(JobVar)

#Save the workflow variables
WfVar = Workflow(jobs = jobs)
somaWF_name = os.path.join(wkdir, 'SomaWF/Decoding/menRot_decVis_workflow')
Helper.serialize(somaWF_name, WfVar)
