#Purpose: Plot decoding analyses for individual subjects.
#Project: MenRot
#Author: Darinka Trubutschek
#Date: 19 December 2017

 plt.figure(figsize=(40,40))

 for sc, subi in enumerate(ListSubjects):
 	plt.subplot(6, 5, sc+1)
 	plt.imshow(scores[sc], origin='lower')
 	plt.title(subi)

 	plt.yticks([ ])
 	plt.xticks([ ])

	plt.colorbar()
 
 plt.savefig(res_path + '/' + my_analysis + '_' + stat_params + str(tail) +  '-' + vis + '-indSub.tif', format = 'tif', dpi = 300, bbox_inches = 'tight')
 plt.show()
