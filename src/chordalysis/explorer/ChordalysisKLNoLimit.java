/*******************************************************************************
 * Copyright (C) 2014 Francois Petitjean
 * 
 * This file is part of Chordalysis.
 * 
 * Chordalysis is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * Chordalysis is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Chordalysis.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package chordalysis.explorer;

import java.util.ArrayList;

import chordalysis.model.ScoredGraphAction;
import chordalysis.tools.ChiSquared;

public class ChordalysisKLNoLimit extends ChordalysisKLMaxNParam {

    public ChordalysisKLNoLimit(long maxNParam) {
	super(30,maxNParam);
    }

    @Override
    protected void explore() {
	long nParamBestModel = bestModel.getNbParameters();
	ArrayList<ScoredGraphAction> rejectedPValue = new ArrayList<>();
	
	while (!pq.isEmpty()) {
	    ScoredGraphAction todo = pq.poll();
	    if (todo.getScore() == Double.POSITIVE_INFINITY) {
		break;
	    }else{
		long dfDiff = bestModel.nbParametersDiffIfAdding(todo.getV1(), todo.getV2());
		double gDiff = 2.0 * this.nbInstances * (1.0/todo.getScore());
		long correctedDfDiff;
		if(dfDiff>1000){
		    correctedDfDiff = (long) (Math.log(dfDiff-100)+1)+100;
		}else if(dfDiff>100){
		    correctedDfDiff = (long) (Math.sqrt(dfDiff-50)+1)+50;
		}else if(dfDiff>50){
		    correctedDfDiff = (dfDiff-25)/2+25;
		}else{
		    correctedDfDiff = dfDiff;
		}
		double pValue = ChiSquared.pValue(gDiff, correctedDfDiff);
		long candidateModelNParam = nParamBestModel+dfDiff;
		if(candidateModelNParam>maxNParam){
		    continue;
		}else if(pValue>.5){
		  //can't add such an edge twice so might as well check p-value
		    System.out.println("evicted "+todo+" with df="+dfDiff+" and p="+pValue);
		    rejectedPValue.add(todo);
		    continue;
		}else{
		    //performing addition
		    operationsPerformed.add(todo);
		    bestModel.performAction(todo, bestModel, pq);
		    nParamBestModel = candidateModelNParam;
		}
	    }
	}
	for(ScoredGraphAction action:rejectedPValue){
	    if(bestModel.graph.isEdgeAddable(action.getV1(), action.getV2())){
		pq.enableEdge(action.getV1(), action.getV2());
	    }
	}
	pq.processStoredModifications();
	System.out.println("second round");
	while (!pq.isEmpty()) {
	    ScoredGraphAction todo = pq.poll();
	    if (todo.getScore() == Double.POSITIVE_INFINITY) {
		break;
	    }else{
		long dfDiff = bestModel.nbParametersDiffIfAdding(todo.getV1(), todo.getV2());
		long candidateModelNParam = nParamBestModel+dfDiff;
		if(candidateModelNParam>maxNParam){
		    continue;
		}else{
		    //performing addition
		    operationsPerformed.add(todo);
		    bestModel.performAction(todo, bestModel, pq);
		    nParamBestModel = candidateModelNParam;
		}
	    }
	}
	
	
	
    }

}
