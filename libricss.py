import numpy as np
path = 'E:/Original Audio/New folder/for_release/for_release/OV10/overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0/transcription/meeting_info.txt'

def convertTimestampsAudacity(input_path, output_path):
    timestamps = np.genfromtxt(input_path, delimiter='\t', dtype=None, encoding=None, names=True, usecols=(0,1))

    ground_truth = np.zeros((timestamps.shape[0]+1, 3))
    ground_truth[0] = [0, timestamps[0][0], 0] 

    for i , (start_time, end_time) in enumerate(timestamps):
        next_start_time = timestamps[i+1][0] if i+1 < len(timestamps) else end_time
        if end_time > next_start_time:
            ground_truth[i+1] = [next_start_time, end_time, 2]
        elif end_time < next_start_time:
            ground_truth[i+1] = [end_time, next_start_time, 0]
        else:
            ground_truth[i+1] = [end_time, next_start_time, 1]

    # export groundtruth as tsv in the same directory as path
    np.savetxt(output_path, ground_truth, delimiter='\t', fmt='%s')


def convertTimestampsRTTM(input_path):
    import numpy as np

    # load the data from the text file
    data = np.genfromtxt(input_path, delimiter='\t', names=True ,usecols=(0,1,2))

    print(data)

    # parse the data into RTTM format
    rttm = []
    for i in range(len(start_times)):
        rttm.append('SPEAKER {} {} {} {} <NA> <NA> {}\n'.format(
            speakers[i],
            start_times[i],
            end_times[i] - start_times[i],
            transcriptions[i]
        ))

    print(rttm)


def main():
    convertTimestampsRTTM('meeting_info.txt')

if __name__ == '__main__':
    main()