package objectiveFunction.parallel;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

//import lbfgsb.FunctionValues;
import optimize.FunctionValues;

import objectiveFunction.ObjectiveFunctionCLL_d;
import EBNC.wdBayes;
import DataStructure.wdBayesNode;

public class ParallelObjectiveFunctionCLL_d extends ObjectiveFunctionCLL_d {

	wdBayesNode[][] nodes;
	int nThreads;
	double[][] gs;
	private double[][] tmpProbs;
	private ExecutorService executor;	

	private static final int minNPerThread = 10000;

	public ParallelObjectiveFunctionCLL_d(wdBayes algorithm) {
		super(algorithm);

		int N = super.algorithm.getNInstances();

		if (N < minNPerThread) {
			this.nThreads = 1;
		} else {
			this.nThreads = Runtime.getRuntime().availableProcessors();
			if (N/this.nThreads < minNPerThread) {
				this.nThreads = N/minNPerThread + 1;
			}
		}

		this.nodes = new wdBayesNode[nThreads][super.algorithm.getnAttributes()];
		this.gs = new double[nThreads][super.algorithm.getdParameters_().getNp()];
		this.tmpProbs = new double[nThreads][super.algorithm.getNc()];
		executor = Executors.newFixedThreadPool(nThreads);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		double negLogLikelihood = 0.0;
		algorithm.getdParameters_().copyParameters(params);
		double g[] = new double[algorithm.getdParameters_().getNp()];

		Future<Double>[] futures = new Future[nThreads];

		int assigned = 0;
		int remaining = algorithm.getNInstances();

		for (int th = 0; th < nThreads; th++) {
			/*
			 * Compute the start and stop indexes for thread th
			 */
			int start = assigned;
			int nInstances4Thread = remaining / (nThreads - th);
			assigned+=nInstances4Thread;
			int stop = assigned-1;
			remaining -= nInstances4Thread;

			/*
			 * Calling thread
			 */
			Callable<Double> thread = new CallableCLL_d(algorithm.getM_Instances(), start, stop, algorithm.getNc(), nodes[th], tmpProbs[th], gs[th], algorithm.getdParameters_(), algorithm.getM_Order(), algorithm);
			futures[th] = executor.submit(thread);
		}

		for (int th = 0; th < nThreads; th++) {
			try {
				negLogLikelihood += futures[th].get();

			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			for (int i = 0; i < g.length; i++) {
				g[i] += gs[th][i];
			}
		}

		return new FunctionValues(negLogLikelihood, g);
	}

	@Override
	public void finish(){
		executor.shutdown();
	}

}