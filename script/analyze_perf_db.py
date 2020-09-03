from __future__ import print_function
import os
import sys
import math
import psycopg2
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.facecolor'] = "0.8"
######################################################
TOTAL_EXECS = 1#16
#samplePeriod=1000


class ModelStat:
	modelName=None
	execStatList=None;
	def __init__(self, modelName, execTypeList):
		self.modelName = modelName;
		self.execStatList = {}
		for execType in execTypeList:
			self.execStatList[execType] = ExecStat(execType, []);
		return

class ExecStat:
	execType=None;
	evtStatList = None;
	execTime = 0;
	execValidCount=0;

	def __init__(self, execType, eventList):
		self.execType = execType;
		self.evtStatList = {};
		for evt in eventList:
			self.evtStatList[evt] = 0;
		return

	def setEventList(self, eventList):
		for evt in eventList:
			self.evtStatList[evt] = EvtStat(evt, 0);
		return

	def getPercentualRatio(self, evtA, evtB):
		evtStatA = self.evtStatList[evtA];
		evtStatB = self.evtStatList[evtB];
		ratio = (evtStatA.evtCount/evtStatB.evtCount)*100;
		return ratio

	def getEvtCount(self, evt):
		return self.evtStatList[evt].evtCount;

	def getEvtPeriod(self, evt):
		return self.evtStatList[evt].avgPeriod;

	def getEvtPTI(self, evt):
		return ((self.evtStatList[evt].evtCount)/(self.getInstructionsCount()/1000));
	
	def getExecIPC(self):
		return (self.getInstructionsCount())/(self.getCPUCLKCount());
	
	def getInstructionsCount(self):
		return self.evtStatList['instructions'].evtCount;

	def getCPUCLKCount(self):
		return self.evtStatList['cpu-clock'].evtCount;

class EvtStat:
	evtName = None;
	evtCount = 0;
	avgPeriod = 0;
	def __init__(self, evtName, evtCount):
		self.evtName = evtName;
		self.evtCount = evtCount;

class DBDataCollector:
	
	fullEventList=None
	chartController=None;
	modelList=None
	execTypeList=None
	hostip = ""

	def __init__(self, modelList, execTypeList, hostip):
		self.fullEventList = []
		self.chartController = ChartController()
		self.modelList = modelList
		self.execTypeList = execTypeList
		self.hostip = hostip

	def getRunTime(self, cursor):
		sql="SELECT time FROM runtime  LIMIT 1"
		#print (sql)
		cursor.execute(sql)
		rows = cursor.fetchall()
		runtime = rows[0][0]
		runtime = int(runtime)
		return runtime

	def getAVGPeriod(self, cursor, evt):
		sql="SELECT id FROM selected_events WHERE name LIKE '%"+evt+"%'  LIMIT 1"
		#print (sql)
		cursor.execute(sql)
		rows = cursor.fetchall()
		evtid = rows[0][0]

		sql="SELECT AVG(period) FROM samples WHERE evsel_id="+str(evtid)+"  LIMIT 1"
		cursor.execute(sql)
		rows = cursor.fetchall()
		avg = rows[0][0]
		return int(avg)

	def getEvtList(self, cursor):
		sql = "SELECT name FROM selected_events WHERE name NOT LIKE '%unknown%'"
		cursor.execute(sql)
		rows = cursor.fetchall()
		evtList = []
		for r in rows:
			evt = str(r[0]).strip()
			if ":u" in evt:
				evt = evt[:-2]
			evtList.append(evt);
		return evtList

	def getEventCount(self, cursor, evt):
		sql="SELECT COUNT(*) FROM samples_view WHERE event LIKE '%" + evt + "%'"
		#print (sql)
		cursor.execute(sql)
		rows = cursor.fetchall()
		return rows[0][0];

	def createModelStatList(self):
		modelStatList=[]
		# iterate in all dbs
		for model in self.modelList:
			# iterate in all execTypes for each model
			mstat = ModelStat(model, execTypeList)
			for execType in self.execTypeList:
				print(" > " + execType + ": ", end="")
				#iterate in all execs for each execType
				count_valid_execs = 0
				for k in range(0, TOTAL_EXECS):
					dbName = model + "_" + execType + "_db" + str(k)
					# connectig to specific db
					try:
						conn = psycopg2.connect("dbname='"+dbName+"' user='fpga' host='"+self.hostip+"' password='123'")
					except:
						print ("Could not connect to: " + dbName)
						continue
					cur = conn.cursor()
					eventList = self.getEvtList(cur)
					mstat.execStatList[execType].setEventList(eventList)
					mstat.execStatList[execType].execTime = self.getRunTime(cur)
					# accumulate event counts
					for evt in eventList:
						evtStat = mstat.execStatList[execType].evtStatList[evt];
						evtStat.evtName = evt;
						evtStat.evtCount = evtStat.evtCount + self.getEventCount(cur, evt);
						evtStat.avgPeriod = evtStat.avgPeriod + self.getAVGPeriod(cur, evt);
						if evt not in self.fullEventList:
							self.fullEventList.append(evt)
						#print (" Model: " + model + " Exec: " + execType + " EVT: " + evt + " VAL:" + str(value))
					conn.close()
					count_valid_execs = count_valid_execs + 1;
				mstat.execStatList[execType].execValidCount = count_valid_execs;
				print("Valid Execs=" + str(count_valid_execs) + " Number of Events="+str(len(eventList)))
			modelStatList.append(mstat)
			# print("")
		self.normalizeModelStatList(modelStatList)
		return modelStatList

	def normalizeModelStatList(self, mstatList):
		for mstat in mstatList:
			for execType in self.execTypeList:
				execStat = mstat.execStatList[execType]
				valid_execs = execStat.execValidCount;
				for evt in execStat.evtStatList:
					evtStat = execStat.evtStatList[evt]
					evtStat.avgPeriod = evtStat.avgPeriod/valid_execs;
					evtStat.evtCount = (evtStat.evtCount/valid_execs)*evtStat.avgPeriod;
					
		return

	def plotMStatAbsValues(self, mstat, outputPath):
		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		for evt in self.fullEventList:
			vec = []
			pti =""
			title = mstat.modelName + "_" + evt + pti
			outputFile  = outputPath + "/" + mstat.modelName + "_" + evt + ".png"
			for execType in self.execTypeList:
				execStat = mstat.execStatList[execType]
				value = execStat.getEvtCount(evt)
				vec.append(value)
			self.chartController.plotBar(title,"Value", self.execTypeList, vec, outputFile)

	def plotCacheMissesPercentualRatio(self, mstat, outputPath):
		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		cmisses_vec = []
		l1misses_vec = []
		for execType in self.execTypeList:
			execStat = mstat.execStatList[execType]
			cmisses_ratio = execStat.getPercentualRatio("cache-misses", "cache-references")
			cmisses_vec.append(cmisses_ratio)
			l1misses_ratio = execStat.getPercentualRatio("L1-dcache-load-misses", "L1-dcache-loads")
			l1misses_vec.append(l1misses_ratio)

		outputFile  = outputPath + "/" + mstat.modelName + "_cmisses_ratio.png"
		self.chartController.plotBar("Cache-Misses Ratio(%)","(%)", self.execTypeList, cmisses_vec, outputFile)

		outputFile  = outputPath + "/" + mstat.modelName + "_l1misses_ratio.png"
		self.chartController.plotBar("Cache-Misses Ratio(%)","(%)", self.execTypeList, l1misses_vec, outputFile)

	def plotEvtPTI(self, mstat, outputPath):
		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		for evt in self.fullEventList:
			vec = []
			pti ="_pti"
			title = mstat.modelName + "_" + evt + pti
			outputFile  = outputPath + "/" + mstat.modelName + "_" + evt + "_pti.png"
			for execType in self.execTypeList:
				execStat = mstat.execStatList[execType]
				value = execStat.getEvtPTI(evt)
				vec.append(value)
			self.chartController.plotBar(title,"PTI", self.execTypeList, vec, outputFile)

	def plotExecTime(self, mstat, outputPath):
		if not os.path.exists(outputPath):
			os.makedirs(outputPath)

		outputFile  = outputPath + "/" + mstat.modelName + "_exectime.png"
		vec = []
		for execType in self.execTypeList:
			execStat = mstat.execStatList[execType]
			value = execStat.execTime
			vec.append(value)
		self.chartController.plotBar("Exec Time (s)","(s)", self.execTypeList, vec, outputFile)

class ChartController:
	style = 'dark_background'
	color = 'y'#'m'
	#style = 'ggplot' # 
	#style = 'fivethirtyeight' 
	#style = 'seaborn'
	def __init__(self):
		pass

	def plotBar(self, title, ylabel, xlabels, values, outputFile):
		width = 1/3.0
		N = len(xlabels)
		ind = np.arange(N)
		plt.style.use(self.style)
		f, ax = plt.subplots(1, sharex=True)
		ax.bar(ind, values, width, color=self.color, align="center")
		ax.set_ylabel(ylabel)
		ax.set_xticks(ind)
		ax.set_xticklabels(xlabels)
		ax.set_title(title)
		ax.ticklabel_format(style='sci', axis='y')
		plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
		plt.xlim([min(ind) - 0.5, max(ind) + 0.5])
		plt.tight_layout()
		plt.ylim([0, max(values) + 0.2*max(values)])
		plt.tight_layout()
		if outputFile is not None:
			plt.savefig(outputFile)

if __name__ == '__main__':
	conn = None
	
	# eventList = ["cpu-clock", "instructions", "cache-references",
	# "cache-misses","L1-dcache-loads","L1-dcache-load-misses",
	# "LLC-loads","LLC-load-misses"]
	modelList = ['mod_295x415']
	execTypeList = ['serial', 'opencl_cpu', 'opencl_pac']
	outputPath = sys.argv[1]
	hostip = "172.19.0.217"
	plt.close('all')
	
	dbctrl = DBDataCollector(modelList, execTypeList, hostip)

	print(" Reading Execution Stats...")
	mstatList = dbctrl.createModelStatList()
	print(" Plotting bar charts...")
	for mstat in mstatList:
		outputPath = outputPath + "/" + mstat.modelName
		dbctrl.plotMStatAbsValues(mstat, outputPath)
		dbctrl.plotEvtPTI(mstat, outputPath)
		dbctrl.plotCacheMissesPercentualRatio(mstat, outputPath)
		dbctrl.plotExecTime(mstat, outputPath)
	
	
	print(" Done")