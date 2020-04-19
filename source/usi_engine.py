import sys
import random
# i'm not import logging,
# to prevent mistype that coding 'logging.debug('sample')',
# instead of 'logger.debug('sample')'
from logging import getLogger, basicConfig, DEBUG, ERROR
import shogi

basicConfig(filename='log/log.log', level=DEBUG, format='%(asctime)s,%(levelname)s,%(process)d,%(thread)d,%(module)s,%(lineno)d,%(funcName)s,%(message)s')
logger = getLogger(__name__)

PROGRAM_NAME = 'genkiball'

class USI:
    def __init__(self):
        logger.debug('start')
        self._board = shogi.Board()
        logger.debug('end')

    def run(self):
        logger.debug('start')
        while True:
            logger.info('wait command')
            cmd_line = input()
            logger.info(cmd_line)
            cmds = cmd_line.split(' ')
            if cmds[0] == 'usi':
                self.send_id_name()
                self.send_option()
                self.send_usiok()
            elif cmds[0] == 'setoption':
                pass
            elif cmds[0] == 'isready':
                self.send_readyok()
            elif cmds[0] == 'usinewgame':
                pass
            elif cmds[0] == 'position':
                if cmds[1] == 'startpos':
                    moves = cmds[3:]
                    self._board.reset()
                    _ = list(map(lambda m: self._board.push_usi(m), moves))
                elif cmds[1] == 'sfen':
                    sfen = cmds[2]
                    self._board.push_usi(sfen)
            elif cmds[0] == 'go':
                if self._board.is_game_over():
                    self.send_bestmove('resign')
                else:
                    best_move = random.choice(list(self._board.legal_moves)).usi()
                    self._board.push_usi(best_move)
                    self.send_bestmove(best_move)
            elif cmds[0] == 'quit':
                break
            elif cmds[0] == 'print':
                print(self._board)
        logger.debug('end')

    def send(self, cmd):
        print(cmd)
        logger.info(cmd)

    def send_id_name(self):
        self.send('id name ' + PROGRAM_NAME)

    def send_option(self):
        pass

    def send_usiok(self):
        self.send('usiok')

    def send_readyok(self):
        self.send('readyok')

    def send_bestmove(self, move):
        self.send('bestmove ' + move)

if __name__ == '__main__':
    logger.debug('start')
    usi = USI().run()
    logger.debug('end')
