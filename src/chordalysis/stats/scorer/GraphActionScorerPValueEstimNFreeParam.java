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
import chordalysis.model.PValueScoredGraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;
import chordalysis.stats.NFreeParamsEstimator;
import chordalysis.tools.ChiSquared;

public class GraphActionScorerPValueEstimNFreeParam extends GraphActionScorer {
    int nbInstances;
    EntropyComputer entropyComputer;
    NFreeParamsEstimator estimator;
    double pValueThreshold;

    public GraphActionScorerPValueEstimNFreeParam(int nbInstances, EntropyComputer entropyComputer, NFreeParamsEstimator estimator,
	    double pValueThreshold) {
	this.nbInstances = nbInstances;
	this.entropyComputer = entropyComputer;
	this.estimator = estimator;
	this.pValueThreshold = pValueThreshold;
    }

    @Override
    public ScoredGraphAction scoreEdge(DecomposableModel model, GraphAction action) {

	Double diffEntropy;
	long dfDiff;
	diffEntropy = model.entropyDiffIfAdding(action.getV1(), action.getV2(), entropyComputer);
	dfDiff = model.nbFreeParametersDiffIfAdding(action.getV1(), action.getV2(), estimator);
	System.out.println(dfDiff+ " - "+model.nbParametersDiffIfAdding(action.getV1(), action.getV2()));
	if (diffEntropy == null) {
	    return new PValueScoredGraphAction(action.getType(), action.getV1(), action.getV2(), 1.0, dfDiff, Double.NaN);
	}
	double gDiff = 2.0 * this.nbInstances * (diffEntropy);
	double pValue = ChiSquared.pValue(gDiff, dfDiff);
	double score;
	if (pValue < pValueThreshold) {
	    score = dfDiff / diffEntropy;
	} else {
	    score = Double.POSITIVE_INFINITY;
	}
	ScoredGraphAction scoredAction = new ScoredGraphAction(action.getType(), action.getV1(), action.getV2(), score);
	return scoredAction;

    }

}
