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
import java.util.Comparator;

import chordalysis.lattice.Lattice;
import chordalysis.model.DecomposableModel;
import chordalysis.model.GraphAction;
import chordalysis.model.PValueScoredGraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;
import chordalysis.stats.MyPriorityQueue;
import chordalysis.stats.scorer.GraphActionScorer;
import chordalysis.stats.scorer.GraphActionScorerEntropy;
import chordalysis.stats.scorer.GraphActionScorerPValue;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * This class searches a statistically significant decomposable model to explain
 * a dataset. See paper
 * "Scaling log-linear analysis to high-dimensional data, ICDM 2013"
 * 
 * @see http://www.tiny-clues.eu/Research/
 */
public class SaturatedChordalysis {

	double pValueThreshold;
	EntropyComputer entropyComputer;
	ArrayList<GraphAction> operationsPerformed;
	MyPriorityQueue pq1,pq2;
	GraphActionScorer scorer1,scorer2;
	ArffReader loader;
	DecomposableModel bestModel;
	Lattice lattice;
	Instances dataset;
	int nbInstances;
	int maxK;
	private int nVariables;
	/**
	 * Default constructor
	 * 
	 * @param pValueThreshold
	 *            minimum p-value for statistical consistency (commonly 0.05)
	 */
	public SaturatedChordalysis(double pValueThreshold) {
		this.pValueThreshold = pValueThreshold;
		this.maxK = maxK;
		operationsPerformed = new ArrayList<GraphAction>();
	}

	/**
	 * Launch the modelling
	 * 
	 * @param dataset
	 *            the dataset from which the analysis is performed on
	 */
	public void buildModel(Instances dataset) {
		this.nbInstances = dataset.numInstances();
		this.dataset = dataset;

		int[] variables = new int[dataset.numAttributes()];
		int[] nbValuesForAttribute = new int[variables.length];
		for (int i = 0; i < variables.length; i++) {
			variables[i] = i;
			nbValuesForAttribute[i] = dataset.attribute(i).numValues();
		}
		this.nVariables = variables.length;
		this.lattice = new Lattice(dataset,false);
		this.entropyComputer = new EntropyComputer(dataset.numInstances(),
				this.lattice);
		this.scorer1 = new GraphActionScorerPValue(nbInstances, entropyComputer);
		
		this.bestModel = new DecomposableModel(variables, nbValuesForAttribute);
		
		this.pq1 = new MyPriorityQueue(variables.length, bestModel, scorer1,new Comparator<ScoredGraphAction>() {
			
			@Override
			public int compare(ScoredGraphAction o1, ScoredGraphAction o2) {
				if (o1 instanceof PValueScoredGraphAction && o2 instanceof PValueScoredGraphAction) {
					PValueScoredGraphAction os1 = (PValueScoredGraphAction) o1;
					PValueScoredGraphAction os2 = (PValueScoredGraphAction) o2;
					double ratio1 = os1.getEntropy()/os1.getNDF();
					double ratio2 = os2.getEntropy()/os2.getNDF();
					
					int res;
					
					res = Double.compare(os2.getEntropy(), os1.getEntropy());
					if (res != 0) return res;
					
					res = Double.compare(os1.getScore(), os2.getScore());
					if (res != 0) return res;
					
					res = Long.compare(os1.getNDF(),os2.getNDF());
					if (res != 0) return res;
				}
				return o1.compareTo(o2);
			}
		});
		for (int i = 0; i < variables.length; i++) {
			for (int j = i + 1; j < variables.length; j++) {
				pq1.enableEdge(i, j);
			}
		}
		pq1.processStoredModifications();

		this.explore();
	}

	/**
	 * @return the Decomposable model that has been built
	 */
	public DecomposableModel getModel() {
		return bestModel;
	}

	private void explore() {
		
		
		while (!pq1.isEmpty()) {
			double correctedPValueThreshold = pValueThreshold/pq1.size();
			ScoredGraphAction todo = pq1.poll();
			
			if(todo.getScore() > correctedPValueThreshold){
				break;
			}
//			System.out.println("performing " + todo);
			operationsPerformed.add(todo);
			bestModel.performAction(todo, bestModel, pq1);

//			 System.out.println(bestModel);
		}
		this.maxK = bestModel.treeWidth();
		
		//saturating
		this.scorer2 = new GraphActionScorerEntropy(nbInstances, entropyComputer, maxK);
		this.pq2 = new MyPriorityQueue(nVariables, bestModel, scorer2);
		while(!pq1.isEmpty()){
			ScoredGraphAction todo = pq1.poll();
			pq2.enableEdge(todo.getV1(), todo.getV2());
		}
		pq2.processStoredModifications();
		while (!pq2.isEmpty()) {
			ScoredGraphAction todo = pq2.poll();
			if(todo.getScore()==Double.POSITIVE_INFINITY){
				break;
			}
			operationsPerformed.add(todo);
			bestModel.performAction(todo, bestModel, pq2);
		}
		
	}

	public void setLoader(ArffReader loader) {
		this.loader = loader;
	}

	
}
