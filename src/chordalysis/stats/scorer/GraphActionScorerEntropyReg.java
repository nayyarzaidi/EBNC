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

package chordalysis.stats.scorer;

import java.util.BitSet;

import chordalysis.model.DecomposableModel;
import chordalysis.model.GraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;
import chordalysis.stats.SquaredL2NormComputer;

public class GraphActionScorerEntropyReg extends GraphActionScorerEntropy {
	protected SquaredL2NormComputer normComputer;
	protected double lambda;
	protected double costRegIndependence;
	public GraphActionScorerEntropyReg(int nbInstances,EntropyComputer entropyComputer, SquaredL2NormComputer l2normComputer,int maxK,double lambda,DecomposableModel independence){
		super(nbInstances,entropyComputer,maxK);
		this.normComputer = l2normComputer;
		this.lambda = lambda;
		this.costRegIndependence = 0.0;
		for(BitSet clique:independence.getCliquesBFS()){
		    costRegIndependence+=normComputer.computeNorm(clique);
		}
		this.costRegIndependence = costRegIndependence*lambda/2.0;
	}

	@Override
	public ScoredGraphAction scoreEdge(DecomposableModel model, GraphAction action) {
		Double score;
		int treeWidthIfAdding = model.treeWidthIfAdding(action.getV1(), action.getV2());
		if(treeWidthIfAdding>maxK){
			score=Double.POSITIVE_INFINITY;
		}else{
		    double kldDiff = model.entropyDiffIfAdding(action.getV1(),action.getV2(), entropyComputer);
		    double reg = lambda/2.0*model.regularizationDiffIfAdding(action.getV1(),action.getV2(), normComputer,false);
//		    System.out.println("kldiff="+kldDiff+"\t- reg="+reg);
		    score = -kldDiff+reg;
		    if(score>0){
			score=Double.POSITIVE_INFINITY;
		    }
		    
		}
		ScoredGraphAction scoredAction = new ScoredGraphAction(action.getType(),action.getV1(), action.getV2(), score);
		return scoredAction;
		
	}

}
