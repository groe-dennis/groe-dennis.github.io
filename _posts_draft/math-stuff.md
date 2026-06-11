# Hypergraphs
Generalization of graphs, where there are hyperedges that can connect multiple nodes.

# Mantroids
Invented to bride the gap between graph theory and linear algebra
-> It was noticed that finding the basis (largest possible set of linearly independent vecotors) and the spanning tree of graphs (the largest possible set of edges that connects all nodes without creating a cycle) is the same thing.

* Algorithmic motivation
    * a greedy algorithm makes the locally optimal choice at each step with the hope of finding the global optimum
    * This sometimes fails spectacularly, i.e. in the case of the traveling salesman algorithm
    * Mathematicians proved that a greedy algorithm is guaranteed to find the absolute perfect, optimal solution if and only if the underlying structure is a matroid.

## Definition
* M=(E,I)
mit

einer Grundmenge E
Familie unabhängiger Mengen I⊆2^E (alle teilmengen die unabhängig sind)

Die Elemente von E können dabei fast alles sein, z.B.
E={Kanten} eines Graphs
E={v1​,v2​,v3​,v4​} Liste von vektoren

(Bei Vektoren ist unabhängig klar, bei Graphen ist unabhängig= kreisfrei, da man zB
A     B
|     |
D --- C
hier kante AB durch AD-DC-CB ersetzen kann, die Kante kann also aus anderen entstehen, so wie auch bei vektoren)

* Austauschaxiom: Hat man zwei unabhängige Mengen I₁ und I₂ mit |I₂| > |I₁|, dann gibt es immer eine Kante e ∈ I₂ \ I₁, die man zu I₁ hinzufügen kann, ohne einen Kreis zu erzeugen.
-> alle maximalen unabhängigen Mengen haben dieselbe Kardinalität
-> garantiert, dass Kruskal (immer die leichteste Kante ohne Kreis nehmen) den globalen Minimalen Spannbaum findet. (ein greedy-fehler, suboptimale wahl, ist theoretisch korrigierbar, indem man eine Kante tauschen kann)
-> Die grobe Idee:
Greedy nimmt zuerst das schwerste/verlockendste Element g.
Jetzt könnte man fragen:
    Was, wenn die optimale Lösung OPT dieses Element gar nicht enthält?
Dann sagt die Matroid-Struktur:
    Kein Problem. Wegen der Austausch-Eigenschaft kann man OPT so umbauen, dass g hineinkommt, ohne die Unabhängigkeit zu verlieren.
Und weil g das schwerste verfügbare Element ist, wird die Lösung dadurch nicht schlechter. -> greedy optimal, das argument ist nicht wörtlich im Algorithmus zu nehmen, sondern nur theoretisch



* Circuit: Die minimalste Menge die abhängig ist, aber wenn man egal was entfernt ist alles unabhängig (also die minimalste Ursache warum eine Abhängigkeit besteht)

* Greedy funktioniert genau dann, wenn die zulässigen zwischenschritte in dem algorithmus ein Matroid bilden
-> deswegen funktioniert Kruksal, weil da bei jedem teilschritt wir eine menge an kanten haben und jede dieser mengen unabhängig ist, weil am ende der entstehende minimale spannbaum auch unabhängig ist
## Abschwächungen
Es gibt Abschwächungen von Mantroids, die nur sagend, dass greedy "ungefähr" optimal ist:

p-Systems
k-extendible systems
greedoids
polymatroids
submodular maximization under matroid constraints
weak submodularity
RIP / restricted isometry property
mutual incoherence

# Greedoids
A superset of Mantroids

* Definiert einen Wurzelknochen, und nur nur solche Kantenmengen zulässig, die von der Wurzel aus einen zusammenhängenden Baum aufspannen (Branching Greedoid)
()



# Traveling Salesman and greedy algos
Greedy makes locally optimal choices that lead to globally poor results. The core issue: choosing the cheapest next step paints you into a corner later.
-> So the greedy algo does not have a global view

For traveling salesman, which is NP-hard, you instead have to do either dynamic programming (exp) or rely on heuristics/simulated annealing/genetic algorithms/ant colony, for approx solution


# Lokale Minima in High-dimensional spaces. 

In sehr hochdimensionalen, nichtkonvexen Optimierungsproblemen sind “schlechte” echte lokale Minima oft viel seltener als Sattelpunkte.

ntuitiv wird es immer schwieriger, dass eine Funktion in allen Richtungen gleichzeitig “nach oben gekrümmt” ist, was für ein lokales Minimum nötig wäre

Gradient Descent bleibt nicht unbedingt an einem schlechten Tal hängen, sondern kann eher an flachen, schlecht gekrümmten Regionen langsam werden.


# Ideas
* Even though problems in general can be NP-hard, if there is a subset of the problem space you care about you can still probably find a polynomial algorithm, given the subset is restricted enogh, without proving P=NP
-> Related that maybe nn learning is NP hard but for the subset of data we care about it's okay

* Mantroid in graphs seems quite related to Ising method?

* Funktionieren größere nn, weil sie "quasi" unabhängig sind durch große dimension und damit greedy (gd) optimal ist, sodass es die beste Lösung findet? 

