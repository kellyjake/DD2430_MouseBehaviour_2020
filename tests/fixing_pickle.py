import pickle
# Script for saving a pickled tracker that i fucked up

tracker_arg = r'e:\ExperimentalResults\20200728\20200728_behaviour2020_iv\20200728_behaviour2020_iv_9829_1\20200728_behaviour2020_iv_9829_1_tracker.p'

f = open(tracker_arg, 'rb')
tracker = pickle.load(f)
f.close()

def deeper(this_dict):
    for _ in range(len(this_dict)):
        K,V = this_dict.popitem()
        if K == '#9826':
            print("Found #9826!:\n")
        newK = K.replace('#','')
        print(F"Replacing \t {K}")
        this_dict[newK] = V
        print(F"with \t\t {newK}")

    for k,v in this_dict.items():
        if type(v) == dict:
            deeper(v)
        elif type(v) == list:
            K,V = this_dict.popitem()
            newK = k.replace('#','')
            this_dict[newK] = []
            for string in v:
                 print(F"Replacing \t {string}")
                 this_dict[newK].append(string.replace('#',''))
                 print(F"with \t\t {string.replace('#','')}")
                 
        elif type(v) == str:
            K,V = this_dict.popitem()
            newK = k.replace('#','')
            print(F"Replacing \t {v}")
            this_dict[newK] = v.replace('#','')
            print(F"with \t\t {this_dict[k]}")

print(tracker)
deeper(tracker)

for k,v in tracker.items():
    print(F"Key: {k} \t Value: {v}")


#f = open(tracker_arg,'wb')

#pickle.dump(tracker, f)

#f.close()