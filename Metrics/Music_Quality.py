import os
import muspy
import math

dataset_eval = [0.5302987662965006, 3.986320228665771, 0.9921825307541308]
evaluate_map = {
                0: 'polyphony_rate',         
                1: 'pitch_entropy', 
                2: 'groove_consistency',
                }

def dist(a, b):
    return abs(a - b) / b


def evaluation(path):
    files = os.listdir(path)
    files.sort()

    all_metrics = [0] * 3


    for i, file in enumerate(files):
        muse_path = os.path.join(path, file)

        music = muspy.read_midi(muse_path)


        # Pitch-related metrics

        all_metrics[0] += muspy.polyphony_rate(music, threshold=2)
        

        all_metrics[1] += muspy.pitch_entropy(music)


        all_metrics[2] += muspy.groove_consistency(music, measure_resolution=4) #


    all_metrics[0] = all_metrics[0] / len(files)
    all_metrics[1] =  all_metrics[1] / len(files)
    all_metrics[2] =  all_metrics[2] / len(files)


    dis = 0
    for i in range(0, len(all_metrics)):
        dis += dist(all_metrics[i], dataset_eval[i])


    return dis, all_metrics


if __name__=="__main__":

    path = "1/0/midi"
    print(evaluation(path))