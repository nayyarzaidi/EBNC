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

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.random.RandomGenerator;

import chordalysis.lattice.Lattice;
import chordalysis.model.DecomposableModel;
import chordalysis.model.GraphAction;
import chordalysis.model.ScoredGraphAction;
import chordalysis.stats.EntropyComputer;
import chordalysis.stats.MyPriorityQueue;
import chordalysis.stats.NFreeParamsEstimator;
import chordalysis.stats.scorer.GraphActionScorer;
import chordalysis.stats.scorer.GraphActionScorerPValueEstimNFreeParam;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * This class searches a statistically significant decomposable model to explain
 * a dataset. See paper
 * "Scaling log-linear analysis to high-dimensional data, ICDM 2013"
 * 
 * @see http://www.tiny-clues.eu/Research/
 */
public class ChordalysisEstimatedNParams {

    double pValueThreshold;
    EntropyComputer entropyComputer;
    NFreeParamsEstimator nFreeParamEstimator;
    ArrayList<GraphAction> operationsPerformed;
    MyPriorityQueue pq;
    GraphActionScorer scorer;
    ArffReader loader;
    DecomposableModel bestModel;
    Lattice lattice;
    Instances dataset;
    int nbInstances;
    private int nVariables;
    private RandomGenerator rg;

    /**
     * Default constructor
     * 
     * @param pValueThreshold
     *            minimum p-value for statistical consistency (commonly 0.05)
     */
    public ChordalysisEstimatedNParams(double pValueThreshold) {
	this.pValueThreshold = pValueThreshold;
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
	this.lattice = new Lattice(dataset, false);
	this.entropyComputer = new EntropyComputer(dataset.numInstances(), this.lattice);
	this.nFreeParamEstimator = new NFreeParamsEstimator(dataset.numInstances(), this.lattice);
	this.scorer = new GraphActionScorerPValueEstimNFreeParam(nbInstances, entropyComputer, nFreeParamEstimator,pValueThreshold);
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
     * Launch the modelling
     * 
     * @param structure
     *            the structure of the dataset which the analysis is performed
     * @param loader
     *            the stream from which the data can be read to be loaded in
     *            memory
     */
    public void buildModel(Instances structure, ArffReader loader, double samplingRate, boolean ignoreLastAttribute) {
	initDataStructures(structure, loader, samplingRate, ignoreLastAttribute);
	for (int i = 0; i < nVariables; i++) {
	    for (int j = i + 1; j < nVariables; j++) {
		pq.enableEdge(i, j);
	    }
	}
	pq.processStoredModifications();
	this.explore();
    }

    protected void initDataStructures(Instances structure, ArffReader loader, double samplingRate, boolean ignoreLastAttribute) {
	System.out.println("Initialising Chordalysis...");
	this.operationsPerformed = new ArrayList<GraphAction>();
	this.nVariables = structure.numAttributes();
	if (ignoreLastAttribute) {
	    this.nVariables--;
	}
	this.dataset = structure;
	int[] variables = new int[nVariables];
	int[] nbValuesForAttribute = new int[variables.length];
	for (int i = 0; i < variables.length; i++) {
	    variables[i] = i;
	    nbValuesForAttribute[i] = structure.attribute(i).numValues();
	}
	System.out.println("Loading dataset...");
	if (loader == null) {
	    this.lattice = new Lattice(structure, false, ignoreLastAttribute);
	} else {
	    try {
		this.lattice = new Lattice(structure, loader, false, samplingRate, ignoreLastAttribute, rg);
	    } catch (IOException e) {
		System.err.println("IO error with data file");
		e.printStackTrace();
	    }
	}
	this.nbInstances = lattice.getNbInstances();
	this.entropyComputer = new EntropyComputer(nbInstances, this.lattice);
	this.nFreeParamEstimator = new NFreeParamsEstimator(nbInstances, this.lattice);
	this.scorer = new GraphActionScorerPValueEstimNFreeParam(nbInstances, entropyComputer, nFreeParamEstimator,pValueThreshold);
//	this.scorer = new GraphActionScorerPValue(nbInstances, entropyComputer);
	this.bestModel = new DecomposableModel(variables, nbValuesForAttribute);
	this.pq = new MyPriorityQueue(variables.length, bestModel, scorer);
    }

    /**
     * @return the Decomposable model that has been built
     */
    public DecomposableModel getModel() {
	return bestModel;
    }

    private void explore() {


	while (!pq.isEmpty()) {
	    ScoredGraphAction todo = pq.poll();

	    if (todo.getScore() == Double.POSITIVE_INFINITY) {
		break;
	    }
	    // System.out.println("performing " + todo);
	    operationsPerformed.add(todo);
	    bestModel.performAction(todo, bestModel, pq);

	    // System.out.println(bestModel);
	}

    }

    public void setLoader(ArffReader loader) {
	this.loader = loader;
    }
    
    public void setRandomGenerator(RandomGenerator rg) {
	this.rg = rg;
    }

}
