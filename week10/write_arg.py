import json
import os

file = "argument.json"

argument_dict = dict()
argument_dict.setdefault("arguments",[])

argument = {"id":1,"size":[9,9],"start":[0,7],"fake":[7,4],"real":[1,0],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":2,"size":[9,9],"start":[5,0],"fake":[3,4],"real":[0,7],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":3,"size":[9,9],"start":[1,1],"fake":[0,6],"real":[6,4],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":4,"size":[9,9],"start":[5,5],"fake":[7,0],"real":[1,2],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":5,"size":[9,9],"start":[0,4],"fake":[7,2],"real":[5,6],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":6,"size":[9,9],"start":[3,0],"fake":[0,0],"real":[8,5],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":7,"size":[9,9],"start":[3,0],"fake":[8,7],"real":[0,0],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":8,"size":[9,9],"start":[4,4],"fake":[8,8],"real":[0,0],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":9,"size":[9,9],"start":[0,0],"fake":[8,8],"real":[4,4],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":10,"size":[9,9],"start":[3,0],"fake":[8,5],"real":[4,4],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":11,"size":[9,9],"start":[2,0],"fake":[0,8],"real":[8,6],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":12,"size":[9,9],"start":[2,0],"fake":[4,4],"real":[8,7],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":13,"size":[9,9],"start":[3,0],"fake":[8,7],"real":[2,1],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":14,"size":[9,9],"start":[0,1],"fake":[6,5],"real":[1,7],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":15,"size":[9,9],"start":[7,0],"fake":[7,7],"real":[1,6],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":16,"size":[9,9],"start":[7,2],"fake":[3,0],"real":[0,3],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":17,"size":[9,9],"start":[3,7],"fake":[4,0],"real":[0,4],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":18,"size":[9,9],"start":[5,6],"fake":[1,1],"real":[6,0],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":19,"size":[15,15],"start":[3,0],"fake":[14,12],"real":[4,4],"wall":[]}
argument_dict["arguments"].append(argument)

argument = {"id":20,"size":[15,15],"start":[6,0],"fake":[0,13],"real":[11,14],"wall":[]}
argument_dict["arguments"].append(argument)


argument_json = json.dumps(argument_dict)
with open(file,'w') as f:
    f.write(argument_json)
