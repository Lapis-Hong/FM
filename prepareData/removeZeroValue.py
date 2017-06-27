def removeZeroValue(originData,resultData):
	import subprocess

	sed_cmd_temp=[]
	sed_cmd_temp.append("sed -e 's/\s*[0-9]*:0//g' ")
	sed_cmd_temp.append(originData)
	sed_cmd_temp.append(" > ")
	sed_cmd_temp.append(resultData)
	sed_cmd = ''.join(sed_cmd_temp)
	print(sed_cmd)

	subprocess.call(sed_cmd, shell=True)
	
	return 0
