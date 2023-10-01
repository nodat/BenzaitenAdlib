import music21.midi
import mido
import numpy as np


# コードのサフィックスを除外
def remove_chord_suffix(chord_string):
    lst = list(chord_string)
    if not chord_has_suffix(chord_string):
        return chord_string
    if lst[1] == "m" and lst[2] != "a":
        return lst[0] + lst[1]
    else:
        return lst[0]


# コードがサフィックスを持っていればTrue、なければFalse
def chord_has_suffix(chord_string):
    lst = list(chord_string)
    if len(lst) <= 2:
        if len(lst) <= 1:
            return False
        if len(lst) == 2 and lst[1] == "m":
            return False
    return True


# ノート補正
def corrected_note_num_list(note_num_list, chord_prog):
    fixed_note_num_list = []
    prev_note = -1
    for i, e in enumerate(note_num_list):
        # オクターブレベル補正
        e = min(max(60, e), 60 + 12 * 4)
        # １オクターブ以上の移動は禁止
        if i != 0 and abs(e - prev_note) > 12:
            e = e + (prev_note % 12 - e % 12)  # 最寄りの音階の移動に修正
        # Maj 7th の降下は禁止
        if i != 0 and (prev_note - e) == 11:
            e = prev_note + 1
        # 音補正
        area_chord = chord_prog[i // 4]  # 1拍ごとにとりだし（4分音符単位のため）
        valid_notes = list(map(lambda x: x.pitch.midi % 12, area_chord._notes))
        # ルートがEとBの場合は9thの差が異なるため注意
        if (area_chord.root().midi % 12 == 4 or area_chord.root().midi % 12 == 11):
            ninth_diff = 1
        else:
            ninth_diff = 2
        # ルートがEとAとBの場合は6thの差が異なるため注意
        if (area_chord.root().midi % 12 == 4 or area_chord.root().midi % 12 == 9 or area_chord.root().midi % 12 == 11):
            sixth_diff = 8
        else:
            sixth_diff = 9
        valid_notes.append((area_chord.root().midi % 12 + ninth_diff) % 12)  # add 9th note
        valid_notes.append((area_chord.root().midi % 12 + sixth_diff) % 12)  # add 6th note
        # よな抜きにするため7th系は対象外
        seventh_note = (area_chord.root().midi + 10) % 12
        if seventh_note in valid_notes:
            valid_notes.remove(seventh_note)  # remove 7th note
        major_seventh_note = (area_chord.root().midi + 11) % 12
        if major_seventh_note in valid_notes:
            valid_notes.remove(major_seventh_note)  # remove Maj7th note
        valid_notes = np.unique(valid_notes)
        fixed_note = fixed_note_num(e, valid_notes)
        print("note: %d %d" % (fixed_note, (fixed_note % 12 if (fixed_note != -1) else -1)) + "|" + str(area_chord._notes) + "|" + str(valid_notes))
        fixed_note_num_list.append(fixed_note)
        if e != -1:
            prev_note = e

    return fixed_note_num_list


def fixed_note_num(note_num, valid_notes):
    for i, e in enumerate(valid_notes):
        if note_num == -1:
            return note_num
        elif note_num % 12 == e:
            return note_num
        elif (e - note_num % 12) > 0:  # はまる音が近くにある上位の音を使う。
            fixed_note = note_num + (e - note_num % 12)
            print("fixed note: %d -> %d %d" % (note_num, fixed_note, fixed_note % 12))
            return fixed_note
    return note_num


def arrange_using_midi(target_midi: music21.midi):
    res_midi = target_midi
    res_main_tml = []

    note_on_t = 0
    note_off_t = 0
    note_value = 0

    bend_chk_counter = 9999

    # 末尾に追加されたトラックに対して行う
    track = target_midi.tracks[-1]

    for msg in track:
        if msg.type == 'note_on':
            note_on_t = msg.time
            note_value = msg.note
        elif msg.type == 'note_off':
            note_off_t = msg.time
            bend_chk_counter += note_off_t
            if note_off_t >= 240 and bend_chk_counter > 1439:
                note_on_msg = mido.Message('note_on', note=note_value, velocity=127, time=note_on_t)
                res_main_tml.append(note_on_msg)
                # -- bend --
                bend_curve = concave_increasing_bend_curve()
                if note_off_t >= 359:
                    bend_curve = convex_increasing_bend_curve()
                for c in bend_curve:
                    bend_msg_on = mido.Message('pitchwheel', channel=0, pitch=c, time=12)
                    res_main_tml.append(bend_msg_on)
                bend_reset_msg = mido.Message('pitchwheel', channel=0, pitch=0, time=0)
                res_main_tml.append(bend_reset_msg)

                note_off_msg = mido.Message('note_off', note=note_value, velocity=127, time=note_off_t - 120)
                res_main_tml.append(note_off_msg)

                bend_chk_counter = 0
            else:
                note_on_msg = mido.Message('note_on', note=note_value, velocity=127, time=note_on_t)
                res_main_tml.append(note_on_msg)
                note_off_msg = mido.Message('note_off', note=note_value, velocity=127, time=note_off_t)
                res_main_tml.append(note_off_msg)

        else:
            res_main_tml.append(msg)
    res_midi.tracks[-1] = res_main_tml
    return res_midi


def convex_increasing_bend_curve():
    curve = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return list(map(lambda x: x - 4097, curve))


def concave_increasing_bend_curve():
    curve = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    return reversed(list(map(lambda x: 0 - x, curve)))


def linear_increasing_bend_curve():
    curve = [400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
    return reversed(list(map(lambda x: 0 - x, curve)))


def make_last_measure_notes(chord_prog, note_num_list):
    last_measure_chord = chord_prog[-1]
    note_num_list = note_num_list + [int(get_last_note(note_num_list) / 12) * 12 + last_measure_chord.root().midi % 12] * 8 + [int(get_last_note(note_num_list) / 12) * 12 + 4] * 8
    print(note_num_list)
    return note_num_list


def get_last_note(note_num_list):
    filter_list = filter(lambda x: x != -1, note_num_list)
    print(list(filter_list))
    if len(list(filter_list)) == 0:
        return 60
    else:
        return list(filter_list)[-1]
