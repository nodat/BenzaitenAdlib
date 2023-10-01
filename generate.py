import benzaitencore as bc
import music_utils as mu
import numpy as np
import datetime
import time


# 処理時間計測
def print_proc_time(f):
    def print_proc_time_func(*args, **kwargs):
        start_time = time.process_time()
        return_val = f(*args, **kwargs)
        end_time = time.process_time()
        elapsed_time = end_time - start_time
        print("FUNCTION: %s (%s sec)" % (f.__name__, elapsed_time))
        return return_val

    return print_proc_time_func


# model_idf = AIモデルの識別子
# remove_suffix_prob = 伴奏の最後の音を削除する確率
# strict_mode = 伴奏の最後の音を削除するかどうか
def generate_adlib_files(model_idf, remove_suffix_prob, strict_mode=False):
    # タイムスタンプ
    timestamp = format(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

    # ファイル定義
    backing_file = "sample/sample_backing.mid"
    chord_file = "sample/sample_chord.csv"

    # config 読み込み
    model_idf = model_idf.replace("_ST", "")
    model_idf = model_idf.replace("_O", "")
    config_file = open(bc.MODEL_DIR + "%s.benzaitenconfig" % model_idf, 'r')
    configurations = config_file.readlines()
    seq_length = int(configurations[0])
    input_dim = int(configurations[1])
    output_dim = int(configurations[2])
    config_file.close()

    # VAEモデルの読み込み
    main_vae = bc.make_model(seq_length, input_dim, output_dim)
    main_vae.load_weights(bc.MODEL_DIR + "mymodel_%s.h5" % model_idf)

    # コード進行の読み込み
    # appending で最後の小節を追加
    chord_prog = bc.read_chord_file(bc.BASE_DIR + chord_file)
    chord_prog_magenta = bc.parse_chord_for_magenta(chord_prog)
    print(chord_prog_magenta)
    # 4分音符区切りのコード進行をもとに、コード進行のベクトルを作成
    chroma_vec = bc.chord_seq_to_chroma(bc.make_chord_seq(chord_prog, bc.N_BEATS))
    # 空のピアノロールを作成
    pianoroll = bc.make_empty_pianoroll(chroma_vec.shape[0])

    # コードベクターに対する時系列予測を行う　（8小節の場合、4小節ずつ予測）
    for i in range(0, bc.MELODY_LENGTH, bc.UNIT_MEASURES):
        o, c = bc.extract_seq(i, pianoroll, chroma_vec)
        x, y = bc.calc_xy(o, c)
        y_new = main_vae.predict(np.array([x]))
        index_from = i * (bc.N_BEATS * bc.BEAT_RESO)
        print(y_new[0].shape)
        print(y_new[0])
        pianoroll[index_from: index_from + y_new[0].shape[0], :] = y_new[0]

    # ノートナンバーの計算
    note_num_list = bc.calc_notenums_from_pianoroll(pianoroll)

    # 補正
    note_num_list = mu.corrected_note_num_list(note_num_list, chord_prog)
    # 最終小節のノートナンバーを追加
    note_num_list = mu.make_last_measure_notes(chord_prog, note_num_list)

    # 同一ノートを結合
    durations, dur_fixed_notes = bc.calc_durations(note_num_list)

    # メロディトラックの作成
    melody_midi = bc.make_melody_midi(note_num_list, durations, bc.make_base_midi(), 12)
    # MIDIファイル補正(ベンドなど表現を調整）
    melody_midi = mu.arrange_using_midi(melody_midi)

    # 伴奏ファイルの読み込み
    backing_mus_path = bc.BASE_DIR + "/" + backing_file
    # メロディつきMIDIデータの作成
    target_midi = bc.make_melody_midi(note_num_list, durations, bc.read_midi_file(backing_mus_path), 12)
    # MIDIファイル補正(ベンドなど表現を調整）
    target_midi = mu.arrange_using_midi(target_midi)

    # サフィックス定義
    suffix = "%s" % model_idf

    # 提出用MIDIファイルを生成（メロディのみ）
    bc.make_midi_for_submission(
        melody_midi,
        bc.BASE_DIR + "output-melody/melody_%s-%s.mid" % (suffix, timestamp)
    )

    # 確認用MIDI/WAVファイルのセーブ (伴奏つき）
    bc.make_midi_and_wav_for_check(
        target_midi,
        bc.BASE_DIR + "output/all_%s-%s.mid" % (suffix, timestamp),
        bc.BASE_DIR + "output-wav/all_%s-%s.wav" % (suffix, timestamp),
    )




@print_proc_time
def generate_file_set():
    generate_adlib_files("C_major", 1)
    generate_adlib_files("A_minor", 1)
    # generate_adlib_files("C_major_O", 1)
    # generate_adlib_files("C_major_O", 0.5)
    #
    # generate_adlib_files("C_major_ST", 0, True)
    # generate_adlib_files("C_major_ST", 1, True)
    # generate_adlib_files("C_major_ST", 0.5, True)


generate_file_set()
