
nagents = 0
ntimesteps = 0

filepath_odometry = "/home/leo/Desktop/project/full_graph.g2o"
filepath_estimation = "/home/leo/Desktop/project/fullGraph_optimized.g2o"
filepath_corrected = "/home/leo/Desktop/project/fullGraph_optimized_corrected.g2o"


#********* Initialization ************************************************
f1 = open(filepath_odometry, "r")
lines_odometry = f1.readlines()
f2 = open(filepath_estimation, "r")
lines_estimation = f2.readlines()

f3 = open(filepath_corrected, "w")

for l in range(1, len(lines_estimation)):
    if int(lines_estimation[l].split(" ")[1]) == 0:
        break
    ntimesteps = l+1

for line in lines_odometry:
    if len(line.split(" ")) > 9:
        break
    nagents = (int(line.split(" ")[1])+1) // ntimesteps

print("There are " + str(nagents) + " agents running for " + str(ntimesteps) + " time steps.")
index_edges = nagents * ntimesteps


#********* Correction of vertices ************************************************
for i in range(index_edges):
    l1 = lines_odometry[i].replace("\n", "")
    l2 = lines_estimation[i].replace("\n", "")

    parts1 = l1.split(" ")
    parts2 = l2.split(" ")

    for part in parts1[0:2] + parts2[2:]:
        f3.write(part + " ")
    f3.write("\n")


#********* Correction of edges ************************************************
intras = []
for a in range(nagents):
    intras.append(lines_odometry[index_edges+a*(ntimesteps-1):index_edges+(a+1)*(ntimesteps-1)])

def sort_func(e):
    parts = e.split(" ")
    return int(parts[1])

inters = lines_odometry[index_edges+nagents*(ntimesteps-1):]
inters.sort(key=sort_func)

measurements = []
inter_index = 0
for a, intra in enumerate(intras):
    measurements += intra
    for i in range(inter_index, len(inters)):
        if int(inters[i].split(" ")[1]) >= ((a+1) * ntimesteps): 
            break
        measurements.append(inters[i])
        inter_index += 1

for i in range(len(measurements)):
    part1 = measurements[i].split(" ")[0:3]
    part2 = lines_estimation[(nagents*ntimesteps)+i].replace("\n", "").split(" ")[3:]

    for part in part1 + part2:
        f3.write(part + " ")
    f3.write("\n")


#********* Closing of files ************************************************
f1.close()
f2.close()
f3.close()
