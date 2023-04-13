import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


valid_symbols = [
    'aa', 'aa_h', 'aa_l', 'aa_m', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 
    'a_h', 'a_l', 'a_m', 'b', 'bi', 'bu', 'c', 'ch', 'chu', 'cu', 
    'd', 'dd', 'dh', 'di', 'du', 'eh', 'eh_h', 'eh_l', 'eh_m', 'el_h', 
    'el_l', 'el_m', 'er', 'er_h', 'er_l', 'er_m', 'ey', 'f', 'ff', 'fu', 
    'g', 'ga', 'ge', 'gg', 'go', 'gu', 'h', 'hh', 'hu', 'ib_h', 
    'ib_l', 'ib_m', 'if_h', 'if_l', 'if_m', 'ih', 'iy', 'i_h', 'i_l', 'i_m', 
    'jh', 'ji', 'jv', 'k', 'kk', 'ku', 'l', 'li', 'lu', 'lv', 
    'm', 'mi', 'mu', 'n', 'ng', 'ng_h', 'ng_l', 'ng_m', 'ni', 'nn_h', 
    'nn_l', 'nn_m', 'nu', 'nv', 'ow', 'oy', 'o_h', 'o_l', 'o_m', 'p', 
    'ph', 'pi', 'pu', 'qi', 'qv', 'r', 'rr', 'ru', 's', 'sh', 
    'shu', 'ssh', 'su', 't', 'th', 'ti', 'tsh', 'tt', 'tu', 'uh', 
    'uw', 'u_h', 'u_l', 'u_m', 'v', 'v_h', 'v_l', 'v_m', 'wu', 'xi', 
    'xv', 'yi', 'yv', 'z', 'zh', 'zhu', 'zu', 'zz', 'zzh', 
    'zzzar_h', 'zzzar_l', 'zzzar_m', 'zzzer_h', 'zzzer_l', 
    'zzzer_m', 'zzznr_h', 'zzznr_l', 'zzznr_m', 'zzzor_h', 
    'zzzor_l', 'zzzor_m', 'zzzur_h', 'zzzur_l', 'zzzur_m', 
]


_pad      = '_'
_special  = '-'
_silences = ['@sp', '@spn', '@sil']
# Prepend "@" to ARPAbet symbols to ensure uniqueness
_arpabet = ['@' + s for s in valid_symbols]
# Export all symbols:
symbols = [_pad] + list(_special) + _arpabet + _silences

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


_break_labels = ['br0', 'br1', 'br2', 'br3', 'br4']
_stress_labels = ['1', '2']
_skip_symbols = _break_labels + _stress_labels

def text_to_sequence(text):
    sequence = []
    for symbol in text.split():
        if symbol in _skip_symbols:
            continue
        symbol = '@' + symbol
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            logger.info("Phoneme %s is not in set. Skip!" % symbol)
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            symbol = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(symbol) > 1 and symbol[0] == '@':
                result += '{%s}' % symbol[1:]
        else:
            logger.info("Id %d is not in set. Skip!" % symbol_id)
    return result.replace('}{', ' ')

