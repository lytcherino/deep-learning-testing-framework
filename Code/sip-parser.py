#!/usr/bin/python

import sys
import re
import abc

labels = ["<Labels to grab>"]


class Parser(object):

    def __init__(self, labels):
        self.labels = labels

    def parse(self, fileName, *regExpr):
        message = self.readFile(fileName)
        return self.extractValues(message, *regExpr);

    def readFile(self, fileName):
        str = ""
        with open(fileName, 'r') as file:
            for line in file:
                str += line

        return str 

    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def extractValues(self, message):
        pass


class SIP_Parser(Parser):

    def __init__(self, labels):
        super(SIP_Parser, self).__init__(labels)

    def extractValues(self, fileContent, *expr):

        # Default expression
        regExpr = '\s*\S*(?<=[ ,]){}=([-]?[A-Za-z0-9]+)'

        if (len(expr) > 0):
            print "Using provided expression\n"
            regExpr = expr
        else:
            print "Using default expression\n"

        branches = 1
        for letter in regExpr:
            if letter == '|':
                branches += 1

        strValues = "";

        for label in self.labels:
            labelledRegExpr = regExpr
            labelledRegExpr = re.sub(r'{}', str(label), labelledRegExpr)

            m = re.search(labelledRegExpr, fileContent)
            for i in range(1, branches + 1):
                if m.group(i) != None:
                    #strValues += str(value + ': ' + m.group(1)) + '\n'
                    strValues += str(m.group(i))
                    if (label != self.labels[-1]):
                        strValues += ', '

        return strValues

if __name__ == "__main__":
    p = SIP_Parser(labels)
    print p.parse(sys.argv[1])
