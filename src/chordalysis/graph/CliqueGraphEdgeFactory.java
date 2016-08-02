package chordalysis.graph;

import java.util.BitSet;

import org.jgrapht.EdgeFactory;

public class CliqueGraphEdgeFactory implements EdgeFactory<BitSet, CliqueGraphEdge> {
	static CliqueGraphEdgeFactory singleton = new CliqueGraphEdgeFactory();
	@Override
	public CliqueGraphEdge createEdge(BitSet arg0,BitSet arg1) {
		return new CliqueGraphEdge(arg0, arg1);
	}
	
	public static CliqueGraphEdgeFactory getInstance(){
		return singleton;
	}

}
