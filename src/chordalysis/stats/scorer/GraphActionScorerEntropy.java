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

import chordalysis.model.DecomposableModel;
import chordalysis.model.GraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;

public class GraphActionScorerEntropy extends GraphActionScorer {
	int nbInstances;
	EntropyComputer entropyComputer;
	int maxK;
	public GraphActionScorerEntropy(int nbInstances,EntropyComputer entropyComputer, int maxK){
		this.nbInstances = nbInstances;
		this.entropyComputer = entropyComputer;
		this.maxK = maxK;
	}

	@Override
	public ScoredGraphAction scoreEdge(DecomposableModel model, GraphAction action) {
		Double score;
		int treeWidthIfAdding = model.treeWidthIfAdding(action.getV1(), action.getV2());
		if(treeWidthIfAdding>maxK){
			score=Double.POSITIVE_INFINITY;
		}else{
			score = 1.0/model.entropyDiffIfAdding(action.getV1(),action.getV2(), entropyComputer);
		}
		ScoredGraphAction scoredAction = new ScoredGraphAction(action.getType(),action.getV1(), action.getV2(), score);
		return scoredAction;
		
	}

}
