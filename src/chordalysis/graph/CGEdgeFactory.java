package chordalysis.graph;

import java.util.TreeSet;

import org.jgrapht.EdgeFactory;

import chordalysis.tools.SortedSets;

public class CGEdgeFactory implements EdgeFactory<TreeSet<Integer>, UniqueTreeSet<Integer>> {
	static CGEdgeFactory singleton = new CGEdgeFactory();
	@Override
	public UniqueTreeSet<Integer> createEdge(TreeSet<Integer> arg0,TreeSet<Integer> arg1) {
		return new UniqueTreeSet<Integer>(SortedSets.intersection(arg0, arg1));
	}
	
	public static CGEdgeFactory getInstance(){
		return singleton;
	}

}
