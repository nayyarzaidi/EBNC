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
package chordalysis.stats;

import java.util.BitSet;
import java.util.HashMap;

import chordalysis.lattice.Lattice;
import chordalysis.lattice.LatticeNode;

/**
 * This class aims at computing multiple entropies between different sets of
 * variables. This class uses different memoizations and memorizations
 * techniques to retrieve results very quickly.
 */
public class NFreeParamsEstimator {
	HashMap<BitSet, Integer> lookup;

	Lattice lattice;
	int nbInstances;

	/**
	 * Constructor
	 * 
	 * @param nbInstances
	 *            number of lines in the database
	 * @param lattice
	 *            associated lattice
	 */
	public NFreeParamsEstimator(int nbInstances, Lattice lattice) {
		this.lookup = new HashMap<BitSet, Integer>();
		this.lattice = lattice;
		this.nbInstances = nbInstances;
		lookup.put(new BitSet(lattice.getNbVariables()), 0);
	}

	public Integer computeNFreeParams(BitSet clique) {
		Integer nFreeParams = lookup.get(clique);
		if (nFreeParams != null) {
			return nFreeParams;
		}

		
		
		
		LatticeNode node = lattice.getNode(clique);
		int nbCells = node.getNbCells();
		int nParents = clique.cardinality();
		if(nParents == 1){
		    return nbCells-1;
		}
		LatticeNode []parents = new LatticeNode[nParents];
		int parentIndex = 0;
		for (int p = clique.nextSetBit(0); p >= 0; p = clique.nextSetBit(p + 1)) {
		    BitSet parentClique = (BitSet) clique.clone();
		    parentClique.clear(p);
		    parents[parentIndex]=lattice.getNode(parentClique);
		    parentIndex++;
		}
		
		
		nFreeParams = 0;
//		 System.out.println("matrix:"+Arrays.toString(matrix));
		int []indexes;
		int [] indexesForParent = new int[nParents];
		
		for (int i = 0; i < nbCells; i++) {
			indexes = node.getIndexes(i);
			if(!node.isMatrixCell0(i)){
			    nFreeParams++;
			    continue;
			}
			
			//look at all possible parents to check if there is a 0 there
			boolean found0 = false;
			for (int p = 0; p < parents.length; p++) {
			    LatticeNode parent = parents[p];
			    
			    //getting associated indexes
			    int index=0;
			    for (int j = 0; j < indexes.length; j++) {
				if(j!=p){
				    indexesForParent[index]=indexes[j];
				    index++;
				}
			    }
			    //now getting count
			    if(parent.isMatrixCell0(indexesForParent)){
				found0 = true;
				break;
			    }
			}
			if(!found0){
			    nFreeParams++;
			}
		}
		nFreeParams--;
		lookup.put(clique, nFreeParams);
		return nFreeParams;
	}

	/**
	 * 
	 * @return the number of lines in the database
	 */
	public int getNbInstances() {
		return nbInstances;
	}

	/**
	 * @return the number of variables in the dataset
	 */
	public int getNbVariables() {
		return lattice.getNbVariables();
	}
	
	public int getSizeLookup(){
		return lookup.size();
	}

}
