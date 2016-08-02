package chordalysis.graph;

import java.util.Collection;
import java.util.TreeSet;

public class UniqueTreeSet<L> extends TreeSet<L> {

	private static final long serialVersionUID = 8083106570463160149L;
	
	public UniqueTreeSet(){
		super();
	}
	
	public UniqueTreeSet(Collection<L> collection) {
		super(collection);
	}

	@Override
	public boolean equals(Object o) {
		if (o == this)
            return true;
		return false;
	}
}
