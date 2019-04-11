使用广度优先搜索的方式让agent获得了一个检验功能，使agent在处于fake goal与real goal之间的一条最优路径内不会学习另一个agent的reward。————你可以说这个是model based，但是至少对于path planning问题，这个思路可以是model free的（把BFS改成Uniform Cost Search可以搞定路径有cost的问题）。

修改了遇到墙的处理方式，变为使agent在撞墙之后会在学习之前重新选择。

加入了一个discount factor。

在可见的未来内，如果墙不在（没有墙时的）最优LDP与goal的路径之间这个系统至少不会在show的时候卡死在某个点了……

cutshow版本中，加入了一个在show的时候如果死循环可以提前终止的机制。意外地解决掉了墙在（没有墙时的）最优LDP与goal的路径之间的问题。

我们似乎可以把精力放在墙在start与最优LDP之间，或者墙正好卡住了最优LDP的情况了。
