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

import chordalysis.lattice.Lattice;
import chordalysis.model.DecomposableModel;
import chordalysis.model.GraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;
import chordalysis.stats.MyPriorityQueue;
import chordalysis.stats.scorer.GraphActionScorer;
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
public class Chordalysis {

	double pValueThreshold;
	EntropyComputer entropyComputer;
	ArrayList<GraphAction> operationsPerformed;
	MyPriorityQueue pq;
	GraphActionScorer scorer;
	ArffReader loader;
	DecomposableModel bestModel;
	Lattice lattice;
	Instances dataset;
	int nbInstances;
	private double minAverageSamplesPerParameter;
	private int maxNumberParametersPerClique;

	/**
	 * Default constructor
	 * 
	 * @param pValueThreshold
	 *            minimum p-value for statistical consistency (commonly 0.05)
	 */
	public Chordalysis(double pValueThreshold) {
		this.pValueThreshold = pValueThreshold;
		this.minAverageSamplesPerParameter = 3.5;
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
		this.lattice = new Lattice(dataset,false);
		this.entropyComputer = new EntropyComputer(dataset.numInstances(),
				this.lattice);
		this.scorer = new GraphActionScorerPValue(nbInstances, entropyComputer);
		this.bestModel = new DecomposableModel(variables, nbValuesForAttribute);

		this.pq = new MyPriorityQueue(variables.length, bestModel, scorer);
		for (int i = 0; i < variables.length; i++) {
			for (int j = i + 1; j < variables.length; j++) {
				pq.enableEdge(i, j);
			}
		}
		pq.processStoredModifications();

		this.explore();
	}

	/**
	 * @return the Decomposable model that has been built
	 */
	public DecomposableModel getModel() {
		return bestModel;
	}

	private void explore() {
		
		this.maxNumberParametersPerClique = (int) (nbInstances/this.minAverageSamplesPerParameter);
		
		
		while (!pq.isEmpty()) {
			double correctedPValueThreshold = pValueThreshold/pq.size();
			ScoredGraphAction todo = pq.poll();
			
			if(bestModel.getNumberParametersCabIfAdding(todo.getV1(), todo.getV2()) >= maxNumberParametersPerClique){
				continue;
			}else if(todo.getScore() > correctedPValueThreshold){
				break;
			}
//			System.out.println("performing " + todo);
			operationsPerformed.add(todo);
			bestModel.performAction(todo, bestModel, pq);

//			 System.out.println(bestModel);
		}
		
		
		
	}

	public void setLoader(ArffReader loader) {
		this.loader = loader;
	}

	
}
