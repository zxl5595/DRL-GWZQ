cutshow版本:

使用广度优先搜索的方式让agent获得了一个检验功能，使agent在处于fake goal与real goal之间的一条最优路径内不会学习另一个agent的reward。————你可以说这个是model based，但是至少对于path planning问题，这个思路可以是model free的（把BFS改成Uniform Cost Search可以搞定路径有cost的问题）。

修改了遇到墙的处理方式，变为使agent在撞墙之后会在学习之前重新选择。

在可见的未来内，如果墙不在（没有墙时的）最优LDP与goal的路径之间这个系统至少不会在show的时候卡死在某个点了……

加入了一个在show的时候如果死循环可以提前终止的机制。意外地解决掉了墙在（没有墙时的）最优LDP与goal的路径之间的问题。（但是不稳定，有可能得到非optimal的路线）

我们似乎可以把精力放在墙在start与最优LDP之间，或者墙正好卡住了最优LDP的情况了。



ql版本：
基于cutshow版本加入了绕墙机制：如果agent看到了身边有墙，则强制采取observer的行动。

基本保证了agent会选择一条强的dissimulation路线。但是同样不是optimal的。




不optimal的现象可能基于同一个原因：observer训练的不稳定性。在我们希望其向着real goal与fake goal的共同方向的时候，observer是有可能选择另一个方向的。然而我们如今的机制并不能修正这一点。

可能的处理方式：一个谨慎选择的discount factor？
