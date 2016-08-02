/*******************************************************************************
 * Copyright (C) 2015 Francois Petitjean
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
public class SquaredL2NormComputer {
	HashMap<BitSet, Double> lookup;

	Lattice lattice;
	int N;

	/**
	 * Constructor
	 * 
	 * @param nbInstances
	 *            number of lines in the database
	 * @param lattice
	 *            associated lattice
	 */
	public SquaredL2NormComputer(int nbInstances, Lattice lattice) {
		this.lookup = new HashMap<BitSet, Double>();
		this.lattice = lattice;
		this.N = nbInstances;
		lookup.put(new BitSet(lattice.getNbVariables()), 0.0);
	}

	/**
	 * Computes the squared l2 norm associated with clique 
	 * 
	 * @param clique
	 *            the lattice node represented by a set of integers
	 * @return the l2norm
	 */
	public Double computeNorm(BitSet clique) {
		Double computedNorm = lookup.get(clique);
		if (computedNorm != null) {
//			System.out.println("cached entropy for clique "+clique+":"+clique.hashCode());
			return computedNorm;
		}
//		System.out.println("Getting entropy for clique "+clique+":"+clique.hashCode());
//		System.out.println("computing entropy for clique "+clique);

		double norm = 0.0;
		LatticeNode node = lattice.getNode(clique);
		int nbCells = node.getNbCells();
//		 System.out.println("matrix:"+Arrays.toString(matrix));
		for (int i = 0; i < nbCells; i++) {
			int O = node.getMatrixCell(i)+1;
			double param = 1.0*O/(N+nbCells);
			norm += param*param;
		}
		lookup.put(clique, norm);
		return norm;
	}

	/**
	 * 
	 * @return the number of lines in the database
	 */
	public int getNbInstances() {
		return N;
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
