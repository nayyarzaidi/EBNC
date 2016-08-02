/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * xyDist.java     
 * Code written by: Nayyar Zaidi
 * 
 */

package DataStructure;

import Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class xyDist {

	private int[][] counts_;
	private int[] classCounts_;	

	private double[][] probs_;
	private double[] classProbs_;

	private int N;
	private int n;
	private int nc;

	private int paramsPerAtt[];	

	public xyDist(Instances instances) {

		N = instances.numInstances();
		n = instances.numAttributes() - 1; // -1 is due to the class presence in numAttributes
		nc = instances.numClasses();

		paramsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			paramsPerAtt[u] = instances.attribute(u).numValues();
		}

		classCounts_ = new int[nc];
		counts_ = new int[n][];

		for (int u1 = 0; u1 < n; u1++) {
			counts_[u1] = new int[paramsPerAtt[u1] * nc];
		}	
	}

	public void addToCount(Instances instances) {		
		for (int ii = 0; ii < N; ii++) {
			Instance inst = instances.instance(ii);
			update(inst);
		}
	}

	public void update(Instance inst) {
		int x_C = (int) inst.classValue();
		classCounts_[x_C]++;		

		for (int u1 = 0; u1 < n; u1++) {
			int x_u1 = (int) inst.value(u1);
			int pos = x_u1*nc + x_C;
			counts_[u1][pos]++;
		}
	}	

	public void countsToProbs() {

		classProbs_ = new double[nc];
		probs_ = new double[n][];

		for (int u1 = 0; u1 < n; u1++) {
			probs_[u1] = new double[paramsPerAtt[u1] * nc];
		}

		for (int c = 0; c < nc; c++) {
			classProbs_[c] = Math.log(SUtils.MEsti(classCounts_[c], N, nc));
		}

		for (int c = 0; c < nc; c++) {
			for (int u = 0; u < n; u++) {				
				for (int uval = 0; uval < paramsPerAtt[u]; uval++) {
					int pos = uval*nc + c;
					probs_[u][pos] = Math.log(Math.max(SUtils.MEsti(counts_[u][pos], classCounts_[c], paramsPerAtt[u]), 1e-75));
				}				
			}
		}

	}
	
	public double[] getClassProbs() {
		return classProbs_;		
	}

	// p(a=v|Y=y) using M-estimate
	public double p(int u1, int u1val, int y) {
		//return mEstimate(counts_[a][v*noOfClasses_+y], classCounts.get(y), paramsPerAtt[a]);
		int pos = u1val*nc + y;
		return SUtils.MEsti(counts_[u1][pos], classCounts_[y], paramsPerAtt[u1]);
	}

	public double pp(int u1, int u1val, int y) {
		//return mEstimate(counts_[a][v*noOfClasses_+y], classCounts.get(y), paramsPerAtt[a]);
		int pos = u1val*nc + y;
		return probs_[u1][pos];
	}

	// p(Y=y) using M-estimate
	public double p(int y) {
		//return (classCounts[y]+M/metaData_->getNoClasses())/(count+M);
		return SUtils.MEsti(classCounts_[y], N, nc); 
	}

	public double pp(int y) {
		//return (classCounts[y]+M/metaData_->getNoClasses())/(count+M);
		return classProbs_[y]; 
	}

	// p(a=v) using M-estimate
	public double p(int u1, int u1val) {
		//return (getCount(a,v)+M/(metaData_->getNoValues(a)))/(count+M);
		return SUtils.MEsti(getCount(u1,u1val), N, paramsPerAtt[u1]);
	}

	// p(a=v, Y=y) using M-estimate
	public double jointP(int u1, int u1val, int y) {
		//return (counts_[a][v*noOfClasses_+y]+M/(metaData_->getNoValues(a)*metaData_->getNoClasses()))/(count+M);
		int pos = u1val*nc + y;
		return SUtils.MEsti(counts_[u1][pos], N, paramsPerAtt[u1] * nc);
	}

	// count[A=v,Y=y]
	public int getCount(int u1, int u1val, int y) {
		//return counts_[a][v*noOfClasses_+y];
		int pos = u1val*nc + y;
		return counts_[u1][pos];
	}

	// count[A=v]
	public int getCount(int u1, int u1val) {
		int c = 0;
		for (int y = 0; y < nc; y++) {
			int pos = u1val*nc + y;
			c += counts_[u1][pos];
		}
		return c;
	}

	// count[Y=y]
	public int getClassCount(int y) { return classCounts_[y]; }

	public int getNoClasses() { return nc; }

	public int getNoAtts() { return n; }

	public int getNoCatAtts() { return n; }

	public int getNoData() { return N; }
	
	public void setNoData() { N++; }

	public int getNoValues(int u) { return paramsPerAtt[u]; }

	public void setClassProbs(int c, double d) {
		classProbs_[c] = d;		
	}

}
