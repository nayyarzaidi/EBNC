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

import chordalysis.model.ScoredGraphAction;

public class ChordalysisKLStopInstancesPerParam extends ChordalysisKLMaxNParam {

    public ChordalysisKLStopInstancesPerParam(long maxNParam) {
	super(30,maxNParam);
    }

    @Override
    protected void explore() {
	long nParamBestModel = bestModel.getNbParameters();
	
	while (!pq.isEmpty()) {
	    ScoredGraphAction todo = pq.poll();
	    if (todo.getScore() == Double.POSITIVE_INFINITY) {
		break;
	    }else{
		long dfDiff = bestModel.nbParametersDiffIfAdding(todo.getV1(), todo.getV2());
		long candidateModelNParam = nParamBestModel+dfDiff;
		long nParamCab = bestModel.getNumberParametersCabIfAdding(todo.getV1(), todo.getV2());
		double samplesPerParam = 1.0*lattice.getNbInstances()/nParamCab;
		
		if(candidateModelNParam>maxNParam){
		    continue;
		}else if(samplesPerParam<1.0){
		  //can't add such an edge twice so might as well check p-value
		    System.out.println("evicted "+todo+" with df="+dfDiff);
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
