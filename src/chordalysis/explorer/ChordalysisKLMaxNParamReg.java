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
import chordalysis.stats.SquaredL2NormComputer;
import chordalysis.stats.scorer.GraphActionScorer;
import chordalysis.stats.scorer.GraphActionScorerEntropyReg;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class ChordalysisKLMaxNParamReg {

    long maxNParam;
    
    int nbInstances;
    private int nVariables;
    private double samplingRate;
    DecomposableModel bestModel;
    EntropyComputer entropyComputer;
    SquaredL2NormComputer l2NormComputer;
    protected Lattice lattice;
    Instances dataset;
    ArrayList<GraphAction> operationsPerformed;
    MyPriorityQueue pq;
    GraphActionScorer scorer;
    int maxK;
    private RandomGenerator rg;
    
    public ChordalysisKLMaxNParamReg(int maxK, long maxNParam) {
	this.maxK = maxK;
	this.maxNParam = maxNParam;
	operationsPerformed = new ArrayList<GraphAction>();
    }


    /**
     * Launch the modelling
     * 
     * @param dataset
     *            the dataset from which the analysis is performed on
     */
    public void buildModel(Instances dataset) {
	this.samplingRate = 1.0;
	initDataStructures(dataset, null, samplingRate, false);
	for (int i = 0; i < nVariables; i++) {
	    for (int j = i + 1; j < nVariables; j++) {
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
	this.samplingRate = samplingRate;
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
		this.lattice = new Lattice(structure, loader, false, samplingRate, ignoreLastAttribute,rg);
	    } catch (IOException e) {
		System.err.println("IO error with data file");
		e.printStackTrace();
	    }
	}
	this.nbInstances = lattice.getNbInstances();
	this.entropyComputer = new EntropyComputer(nbInstances, this.lattice);
	this.l2NormComputer = new SquaredL2NormComputer(nbInstances, this.lattice);
	this.bestModel = new DecomposableModel(variables, nbValuesForAttribute);
	this.scorer = new GraphActionScorerEntropyReg(nbInstances, entropyComputer, l2NormComputer,maxK,0.01,bestModel);
	this.pq = new MyPriorityQueue(variables.length, bestModel, scorer);
    }

    /**
     * @return the Decomposable model that has been built
     */
    public DecomposableModel getModel() {
	return bestModel;
    }


    public void setRandomGenerator(RandomGenerator rg) {
	this.rg = rg;
    }
    

    
    protected void explore() {
	long nParamBestModel = bestModel.getNbParameters();
	
	while (!pq.isEmpty()) {
//	     System.out.println(pq);
	    ScoredGraphAction todo = pq.poll();
	    System.out.println(todo);
	    if (todo.getScore() == Double.POSITIVE_INFINITY) {
		break;
	    }else{
		long candidateModelNParam = nParamBestModel+bestModel.nbParametersDiffIfAdding(todo.getV1(), todo.getV2());
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
