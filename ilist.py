# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 18:30:14 2022

@author: a179227
"""
import json, xarray, csv
from itertools import product
from copy import copy, deepcopy
#import os
import numpy as np
from datetime import datetime

#os.chdir('C:/Users/a179227/OneDrive - Alliance/perso Wx/ES standard/python ESstandard/ES')
#from ESObservation import Observation
#from ESValue import ResultValue, ESValue
#from ESObs import ESObs
#from ESconstante import ES
#from ESSet import ESSet

def identity(*args, **kwargs):
    if len(args) >0 : return args[0]
    elif len(kwargs) >0 : return kwargs[list(kwargs.keys())[0]]
    else : return None

class Ilist:
    '''
    ilist properties :
        
        - lencomplete : number of values if complete (prod(idxlen,not coupled))
        - lenidx : number of index
        - dimension : number of index non coupled and non unique
        - consistent : only one result for one list of index values
        - complete : len == lencomplete and consistent
        - rate : len / lencomplete

    index properties :
        
        - idxlen : len(set(index))
        - idxref : lower coupled index
        - idxunique : idxlen == 1
        - idxcoupled : idxref == iidx[idx]
        
    methods :
        
        - full : all index non coupled and non unique are complete
        - sort : calculate a new list order
        - reorder : apply a new list order
        - json : string json

    static methods :
        
        - _idxlink, _coupled, _transpose, _tuple, _mul, _reorder, _toint, _toext
    '''
    __slots__ = 'extval', 'iidx', 'setidx', 'valname', 'idxname'

    @classmethod 
    def Idict(cls, dictvaliidx, dictsetidx, order=[], idxref=[], defaultidx=True):
        '''valname = 'value'
        validx = []
        if len(dictvaliidx) == 1 :
            valname = list(dictvaliidx.keys())[0]
            validx = dictvaliidx[valname]
        setidx = []
        idxname = []
        for k,v in dictsetidx.items() :
            idxname.append(k)
            setidx.append(v)
        (val, iidx) = Ilist._initset(validx, setidx, idxref, order)'''
        
        return cls(*Ilist._initdict(dictvaliidx, dictsetidx, order, idxref), defaultidx)
     
    @classmethod 
    def Iset(cls, valiidx, setidx, order=[], idxref=[], valname='value', idxname=[], defaultidx=True):
        if   type(valiidx) != list : validx = [valiidx]
        else : validx = valiidx
        #if validx == [] : raise IlistError("valiidx not a list")
        (val, iidx) = Ilist._initset(validx, setidx, idxref, order)
        return cls(val, setidx, iidx, valname, idxname, defaultidx)

    @classmethod 
    def Iindex(cls, val, idxlen, idxref=[], order=[], valname='value', idxname=[], defaultidx=True):
        iidx = Ilist._initindex(len(val), idxlen, idxref, order, defaultidx)       
        return cls(val, [], iidx, valname, idxname, defaultidx)

    @classmethod 
    def Ilist(cls, val=[], extidx=[], valname='value', idxname=[], defaultidx=True):
        if type(val) != list and type(extidx) != list : return
        if defaultidx and (type(extidx) != list or extidx == []) : 
            ext = [list(range(len(val)))]
            idxname = ['default index']
        elif extidx != [] and type(extidx[0]) != list : ext = [extidx]
        else :                                          ext = extidx
        setidx = [Ilist._toset(ind) for ind in ext]
        iidx   = [Ilist._toint(ind, setind) for (ind, setind) in zip(ext, setidx)]
        return cls(val, setidx, iidx, valname, idxname, defaultidx)
                    
    @classmethod 
    def Zip(cls, *args):
        return cls.Ilist([], list(args), 'value',  [], True)

                    
    @classmethod 
    def from_csv(cls, filename='ilist.csv', valfirst=False, order=[], header=True):
        val=[]
        extidx=[]
        valname='value'
        idxname=[]
        defaultidx=True
        with open(filename, newline='') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            first=True
            for row in reader:
                if first : 
                    for vrow in row[1:] : extidx.append([])
                if first and header :
                    if valfirst      : 
                        valname = row[0]
                        for vrow in row[1:] : idxname.append(vrow)
                    else :
                        valname = row[len(row)-1]
                        for vrow in row[:len(row)-1] : idxname.append(vrow)
                else : 
                    if valfirst      : 
                        val.append(row[0])
                        for i in range(len(row)-1) : extidx[i].append(row[i+1])
                    else :
                        val.append(row[len(row)-1])
                        for i in range(len(row)-1) : extidx[i].append(row[i])
                first = False
            return cls.Ilist(val, extidx, valname, idxname, defaultidx)

    def __init__(self, val=[], setidx=[], iidx=[], valname='value', idxname=[], defaultidx=True):
        if   type(val)  != list and type(iidx) != list : return
        if   defaultidx and (type(iidx) != list or iidx == []) : 
            iindex = [list(range(len(val)))]
            if idxname == [] :                      idxname = ['default index']
        elif iidx != [] and type(iidx[0]) != list : iindex = [iidx]
        else :                                      iindex = iidx
        if   type(setidx) != list or setidx == [] : setindex = [Ilist._toset(idx) for idx in iindex]
        elif type(setidx[0]) != list :              setindex = [setidx]
        else :                                      setindex =  setidx
        if type(val) != list                      : extval = [val]
        elif val == [] and iindex == []           : extval = []
        elif val == [] and type(iindex[0]) == list: 
                                                    extval = [True for idx in iindex[0]]
                                                    valname = 'default value'
        else :                                      extval = val
        '''    #if type(iidx[0]) != list : return
            if type(val) != list  :   extval = [val]
            else                    :   extval = [True for idx in iindex[0]]
        else                        :   extval = val'''
        self.extval  = extval
        self.iidx    = iindex
        self.setidx  = setindex
        self.valname = valname
        self.idxname = ['idx' + str(i) for i in range(max(len(iindex), len(setindex)))]
        if type(idxname) == list : 
            for i in range(len(idxname)) :
                if type(idxname[i]) == str : self.idxname[i] = idxname[i]        
        
    @staticmethod
    def _initset(validx, setidx, idxref, order, defaultidx=True):
        if validx == [] :
            val = []
            #iidx = []
            iidx = [list() for i in setidx]
        elif type(validx[0]) != list :
        #if validx == [] or type(validx[0]) != list :
            val = validx
            idxlen = [len(ind) for ind in setidx]
            iidx = Ilist._initindex(len(val), idxlen, idxref, order, defaultidx)       
        else :
            tvalidx = Ilist._transpose(validx)
            val = tvalidx[0]
            iidx = Ilist._transpose(tvalidx[1])
        return (val, iidx)
    
    @staticmethod 
    def _initdict(dictvaliidx, dictsetidx, order=[], idxref=[]):
        valname = 'value'
        validx = []
        if len(dictvaliidx) == 1 :
            valname = list(dictvaliidx.keys())[0]
            validx = dictvaliidx[valname]
        setidx = []
        idxname = []
        for k,v in dictsetidx.items() :
            if type(v) == list : vl = v
            else : vl = [v]
            idxname.append(k)
            setidx.append(vl)
        (val, iidx) = Ilist._initset(validx, setidx, idxref, order)
        return (val, setidx, iidx, valname, idxname)      

    @staticmethod
    def _initindex(vallen, idxlen, idxref, order, defaultidx=True):
        if idxlen == [] :       return []
        if idxlen == [0,0,0] :  return [ [], [], [] ]
        if min(idxlen) == 0 :   return []
        if idxref == [] and defaultidx : ref = list(range(len(idxlen)))
        else : ref = idxref
        if order == [] : orde = list(range(len(idxlen)))   # !!!
        else : orde = order
        if not len(idxlen) == len(ref) : raise IlistError("idxref and idxlen should have the same lenght")
        idxcoupled = [i for i in range(len(idxlen)) if ref[i] != i]
        axes=list(set(ref))
        axesord = []
        for i in orde : 
            #iref = idxref[i]
            iref = ref[i]
            if iref in axes and iref not in axesord : axesord.append(iref)
        for i in axes : 
            if i not in axesord : axesord.append(i)
        iiord = [list(range(idxlen[i])) for i in axesord]
        iidxord = Ilist._transpose(list(product(*iiord)))
        if vallen != len(iidxord[0]) : raise IlistError("val lenght is not consistent with index lenght")
        iidx = list(range(len(iidxord)))
        for idx in axes : iidx[axes.index(idx)] = iidxord[axesord.index(idx)]
        for i in idxcoupled :
            iidx.insert(i,iidx[ref[i]])
        return iidx

    def __repr__(self):
        texLis = ''
        for idx in self.iidx : texLis += '\n' + json.dumps(idx)
        return json.dumps(self.ival) + '\n' + texLis

    def __str__(self):
        return self.json(json_mode='vv', json_string=True)
        texLis = ''
        for (idx, nam) in zip(self.extidx, self.idxname) : 
            #texLis += '\n' + nam + ' : ' + json.dumps(idx)
            texLis += '\n' + nam + ' : ' + self._json(idx)
        return self.valname + ' : ' + self._json(self.extval) + '\n' + texLis
        #return self.valname + ' : ' + json.dumps(self.extval) + '\n' + texLis

    def __eq__(self, other): 
        try: return self.extval == other.extval and self.extidx == other.extidx
        except: return False
               
    def __len__(self): return len(self.extval)

    def __contains__(self, item): return item in self.extval

    def __getitem__(self, ind): return self.extval[ind]

    def __setitem__(self, ind, val): 
        if ind < 0 or ind >= len(self) : raise IlistError("out of bounds")
        self.extval[ind] = val
        self.__init__(self.extval, self.extidx)

    def __add__(self, other):
        ''' Add other's values to self's values in a new Ilist'''
        newilist = self.__copy__()
        newilist.__iadd__(other)
        return newilist

    def __iadd__(self, other):
        ''' Add other's values to self's values'''
        return self.iadd(other, unique=True)

    def __or__(self, other):
        ''' Add other's index to self's index in a new Ilist'''
        newilist = self.__copy__()
        newilist.__ior__(other)
        return newilist

    def __ior__(self, other):
        ''' Add other's index to self's index'''
        if len(self) != len(other) : raise IlistError("the sizes are not equal")
        for i in range(other.lenidx):
            if other.idxname[i] not in self.idxname :
                self.addlistidx(other.idxname[i], other.setidx[i], other.iidx[i])
        return self
    
    def iadd(self, other, unique=False):
        ''' Add other's values to self's values'''
        if self.lenidx != other.lenidx : raise IlistError("the index lenght have to be equal")
        for val,extidx in zip(other.extval, other.textidx) :
            self.append(val, extidx, unique)
        for i in range(self.lenidx) :
            if self.idxname[i][0:3] == 'idx' and other.idxname[i][0:3] != 'idx' : 
                self.idxname[i] = other.idxname[i]
        if self.valname[0:5] == 'value' and other.valname[0:5] != 'value' : 
            self.valname = other.valname
        return self

    def __copy__(self):
        ''' Copy all the data'''
        return deepcopy(self)
    
    @property
    def axes(self): 
        axe = list(filter(lambda x: x >= 0, set(list(map(
                                lambda x,y,z : -1 if (x <= 1 or y) else z, 
                                self.idxlen, self.idxcoupled, self.idxref)))))
        if self.lenidx > 0 and axe == [] : axe = [0]
        return axe

    @property
    def axeslen(self): return [self.idxlen[axe] for axe in self.axes]

    @property
    def complete(self): return self.lencomplete == len(self) and self.consistent
    
    @property
    def consistent(self): 
        # doublon d'index
        return len(set(self._tuple(self.tiidx))) == len(self)

    @property
    def dimension(self): return len(self.axes)

    @property
    def extidx(self): 
        if len(self.iidx) == 0 : return []
        return [ [self.setidx[i][self.iidx[i][j]] for j in range(len(self.iidx[i]))] 
                for i in range(len(self.iidx))]
                
    @property
    def idxcoupled(self): return [self.idxref[i] != i for i in range(self.lenidx)]
    
    @property
    def idxlen(self): 
        return [len(seti) for seti in self.setidx]
        #if self.lenidx == 0 or len(self) == 0 : return []
        #else : return [max(idx)+1 for idx in self.iidx]

    @property
    def idxref(self):
        lis = list(range(self.lenidx))
        for i in range(1,len(lis)):
            for j in range(i):
                if self._coupled(self.iidx[i], self.iidx[j]): 
                    lis[i] = j
                    break
        return lis

    @property
    def idxunique(self): return [ idl == 1 for idl in self.idxlen]

    @property
    def ind(self): return self._transpose(self.extidx)

    @property
    def ival(self): return self._toint(self.extval, self.setval)
    #def ival(self): return self._toint(self.extval, self._toset(self.extval))
    
    @property
    def setval(self): return self._toset(self.extval)
    
    @property
    def lencomplete(self): 
        return self._mul(list(map(lambda x,y: max((1-x)*y,1), self.idxcoupled, self.idxlen)))     

    @property
    def lenidx(self): return len(self.iidx)
        
    @property 
    def minMaxIndex(self):
        if self.lenidx == 0 or len(self) == 0 : return [None, None]
        maxInd = minInd = self.extidx[0][0]
        for idx in self.extidx: 
            minInd = min(minInd, min(idx))
            maxInd = max(maxInd, max(idx))
        return [minInd, maxInd]
    
    @property
    def rate(self): return len(self) / self.lencomplete

    @property
    def setvallen(self): return len(self.setval)

    @property
    def tiidx(self): return self._transpose(self.iidx)
            
    @property
    def textidx(self): return self._transpose(self.extidx)

    @property
    def zip(self): return tuple(tuple(idx) for idx in self.textidx)
             
    def addlist(self, idxname, extidx):
        self.addlistidx(idxname, self._toset(extidx), self._toint(extidx, self._toset(extidx)))
        
    def addlistidx(self, idxname, setidx, iidx):
        if type(iidx) != list or len(iidx) != len(self): 
            raise IlistError("listindex out of bounds")
        if self.lenidx == 1 and self.iidx[0] == list(range(len(self))) :
            self.setidx[0]  = setidx
            self.iidx[0]    = iidx
            self.idxname[0] = idxname
        else : 
            self.setidx .append(setidx)
            self.iidx   .append(iidx)            
            self.idxname.append(idxname)
    
    def append(self, val, extidx, unique=False):
        # unique : check if index is unique
        if not ((self.isIndex(extidx) and unique) or (self.isValue(val) and self.isIndex(extidx))): 
            self.extval.append(val)
            #iidx = self._updateset(idx)
            tiidx = self.tiidx
            #tiidx.append(iidx)
            tiidx.append(self._updateset(extidx))
            self.iidx = self._transpose(tiidx)

    def appendi(self, val, iidx, unique=False):
        # unique : check if index is unique
        if not ((self.isiIndex(iidx) and unique) or (self.isValue(val) and self.isiIndex(iidx))): 
            self.extval.append(val)
            tiidx = self.tiidx
            tiidx.append(iidx)
            self.iidx = self._transpose(tiidx)
      
    def full(self, minind=True, fillvalue=None, inplace=False):
        if not self.consistent : raise IlistError("unable to generate full Ilist with inconsistent Ilist")
        tiidxfull = self.tiidx
        extvalfull = copy(self.extval)
        for ind in self._transpose(self._idxfull(minind)) : 
            if ind not in tiidxfull : 
                tiidxfull.append(ind)
                extvalfull.append(self._nullValue(type(extvalfull[0]), fillvalue))
        iidxfull = self._transpose(tiidxfull)
        if inplace : 
            self.extval = extvalfull
            self.iidx   = iidxfull
        else : return Ilist(extvalfull, self.setidx, iidxfull, self.valname, self.idxname)
        
    def extidxtoi(self, extind):
        try :  return [self.setidx[i].index(extind[i]) for i in range(len(extind))]
        except : return None

    def indidxtoext(self, intind):
        try :  return [self.setidx[i][intind[i]] for i in range(len(intind))]
        except : return None
        
    def iloc(self, index):
        #tiidx = self._transpose(self.iidx)
        try : 
            ival = self.tiidx.index(index)
        except :
            raise IlistError('index not found')
        return self.extval[ival]

    def isIndex(self, extindex):
        iindex = self.extidxtoi(extindex)
        if iindex != None and iindex in self.tiidx : return True
        return False

    def isiIndex(self, iindex):
        if iindex != None and iindex in self.tiidx : return True
        return False

    def isValue(self, value):
        for val in self.extval :
            if value == val : return True
        return False
    
    def json(self, **option):
        option2 = {'json_string' : False, 'json_res_index' :True, 'json_mode' :'vv'} | option
        lis = []
        textidx = []
        if   option2['json_mode'][1]=='v' and self.extidx == [] : 
            lis = [self.valname]
        elif option2['json_mode'][1]=='v' and self.extidx != []: 
            lis=[[self.valname, self.idxname]]
            textidx = self._transpose(self.extidx)
        for i in range(len(self)):
            if option2['json_mode'][0]=='v' : jval = self._json(self.extval[i])
            else            : jval = self.ival[i]
            if not option2['json_res_index'] or self.tiidx == [] : lis.append(jval)
            else :
                lig=[jval]           
                if option2['json_mode'][1]=='v' : lig.append(textidx[i])
                else                            : lig.append(self.tiidx[i])
                lis.append(lig)
        if option2['json_mode'][1]=='v': js = lis
        else : 
            js = {}
            for i in range(len(self.setidx)) :  
                if self.idxlen[i] > 0 : js[self.idxname[i]] = self.setidx[i]
            if len(lis) > 1 : js[self.valname] = lis
        if option2['json_string']: return json.dumps(js)
        else: return js

    def loc(self, extindex):
        iindex = self.extidxtoi(extindex)
        try : 
            ival = self.tiidx.index(iindex)
        except :
            raise IlistError('index not found')
        return self.extval[ival]
    
    def reindex(self, index = []):
        if index == [] : index = list(range(len(self.setidx)))
        for ind in index :
            oldidx = self.setidx[ind]
            self.setidx[ind] = sorted(self.setidx[ind])
            self.iidx[ind]   = Ilist._reindexidx(self.iidx[ind], oldidx, self.setidx[ind])
            
    def reorder(self, idx=[], inplace=True):
        extval = self._reorder(self.extval, idx)
        iidx =  [self._reorder(index, idx) for index in self.iidx]
        if inplace :  
            self.extval = extval
            self.iidx   = iidx
        else : 
            return Ilist(extval, self.setidx, iidx, self.valname, self.idxname)
                      
    def sort(self, sort=[], order=[], reindex=True, inplace=True): 
        return self.reorder(self.sortidx(order, sort, reindex), inplace)

    def sortidx(self, order=[], sort=[], reindex=True):
        #order = [] : tri suivant valeurs
        #sort = ival
        if sorted(sort) == sorted(self.extval): return sort
        newsetidx = [sorted(self.setidx[ind]) for ind in range(len(self.setidx))]
        newiidx = [Ilist._reindexidx(self.iidx[ind], self.setidx[ind], newsetidx[ind]) for ind in range(len(self.setidx))]
        if reindex :
            self.iidx = newiidx
            self.setidx = newsetidx
        if len(order) < 1 or len(order) > self.lenidx or max(order) >= self.lenidx : idx = []
        else : idx = [newiidx[ind] for ind in order]
        idx.append(self.extval)
        idx.append(list(range(len(self))))
        return  self._transpose(sorted(self._transpose(idx)))[len(order)+1]

    def swapindex(self, order):
        if type(order) != list or len(order) != self.lenidx : raise IlistError("order lenght not correct")
        iidx    = []
        setidx  = []
        idxname = []
        for i in order :
            iidx.append  (self.iidx  [i])
            setidx.append(self.setidx[i])
            idxname.append(self.idxname[i])
        self.iidx   = iidx
        self.setidx = setidx
        self.idxname = idxname
            
    def to_csv(self, filename='ilist.csv', func=None, ifunc=[], valfirst=False, order=[], header=True, **kwargs):
        if order == [] : order = list(range(self.lenidx))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, **kwargs)
            if header: 
                row = []
                if valfirst      : row.append(self.valname)
                for idx in order : row.append(self.idxname[idx])
                if not valfirst  : row.append(self.valname)
                writer.writerow(row)
            for i in range(len(self)):
                row = []
                if len(ifunc) == len(self) : funci = func[i]
                else : funci = None
                if valfirst      : row.append(self._funclist(self.extval[i], func, **kwargs))
                for idx in order : row.append(self._funclist(self.extidx[idx][i], funci, **kwargs))
                if not valfirst  : row.append(self._funclist(self.extval[i], func, **kwargs))
                writer.writerow(row)                
  
    def to_numpy(self, func=None, ind='axe', squeeze=True, fillvalue='?', **kwargs):
        # ind : axe, all, flat
        if not self.consistent : raise IlistError("unable to generate numpy array with inconsistent Ilist")
        if   type(ind) == str and ind == 'flat' : 
            return self._tonumpy(self.extval, func=func, **kwargs)
        elif type(ind) == str and ind in ['axe', 'all']  :
            ilf = self.full(minind=ind=='axe', fillvalue=fillvalue)
            ilf.sort(order=self.axes)
            return self._tonumpy(ilf.extval, func=func, **kwargs).reshape(ilf.axeslen)

    def to_xarray(self, info=False, ind='axe', fillvalue='?', func=identity, funcidx=[],
                  name='Ilist', **kwargs):
        if not self.consistent : raise IlistError("Ilist not consistent")
        ilf = self.full(minind=ind=='axe', fillvalue=fillvalue)
        ilf.sort(order=self.axes)
        coord = ilf._xcoord(funcidx, **kwargs)
        dims = [ilf.idxname[ax] for ax in ilf.axes]
        data = ilf._tonumpy(ilf.extval, func=func, **kwargs).reshape(ilf.axeslen)
        if info : return xarray.DataArray(data, coord, dims, attrs=ilf._dict(), name=name)
        else    : return xarray.DataArray(data, coord, dims, name=name)

    def updateidx(self, indval, newextidx, unique=False):
        if len(newextidx) != self.lenidx or indval < 0 or indval >= len(self) :
            raise IlistError("out of bounds")
        if self.isIndex(newextidx) and unique : return
        newiidx = self._updateset(newextidx)
        tiidx = self.tiidx
        tiidx[indval] = newiidx

    def updatelist(self, listidx, idx=None):
        # idx=-1 -> val
        listindex, index = self._checkidxlist(listidx, idx)
        for ind in listindex :
            if len(ind) != len(self) : raise IlistError("out of bounds")
        for (ind, lis) in zip(index, listindex) : 
            if ind == -1 :  self.extval      = lis
            else :
                self.setidx[ind] = self._toset(lis)
                self.iidx[ind] = self._toint(lis, self.setidx[ind])

    def vlist(self, *args, func=identity, idx=None, **kwargs):
        # idx=-1 -> val        
        index = self._checkidxlist(None, idx)[0]
        extidx = self.extidx
        if index == -1 : return self._funclist(self.extval, func, *args, **kwargs)
        else : return self._funclist(extidx[index], func, *args, **kwargs)
   
    def _checkidxlist(self, listidx, idx):      # !!!
        # idx=-1 -> val
        if type(idx) == int : index = [idx]
        elif idx == None :    index = [-1]
        else :                index =  idx
        if min(index) < -1 or max(index) >= self.lenidx : raise IlistError("index out of bounds")
        if listidx == None : return index
        if type(listidx) != list : raise IlistError("listindex not a list")
        if type(listidx[0]) != list :   listindex = [listidx]
        else :                          listindex =  listidx
        if len(index) != len(listindex) : raise IlistError("listindex out of bounds")
        return (listindex, index)
    
    @staticmethod
    def _coupled(ref, l2): return len(Ilist._idxlink(ref, l2)) == len(set(ref))   

    def _dict(self, addproperty=True, long=False):
        dic = {'valname' : self.valname, 'idxname' : self.idxname}
        if long : 
            dic |= {'extval' : self.extval, 'iidx' : self.iidx, 'setidx' : self.setidx }
        if addproperty : 
            dic |= {'axes':self.axes, 'axeslen':self.axeslen, 'complete':self.complete, 
            'consistent':self.consistent, 'dimension':self.dimension,
            'idxcoupled':self.idxcoupled, 'idxlen':self.idxlen, 'idxunique':self.idxunique,
            'idxref':self.idxref, 'lencomplete':self.lencomplete, 'lenidx':self.lenidx,
            'minMaxIndex':self.minMaxIndex, 'rate':self.rate, 'setvallen':self.setvallen }
            if long :
                dic |= { 'extidx':self.extidx, 'ind':self.ind, 'ival':self.ival,
                        'setval':self.setval, 'tiidx':self.tiidx, 'textidx':self.textidx}
        return dic
        
    @staticmethod
    def _funclist(val, func, *args, **kwargs): 
        if func == None or func == [] : return val
        lis = []
        if type(val) != list : listval = [val]
        else : listval = val
        for val in listval :
            try : lis.append(val.func(*args, **kwargs))
            except : 
                try : lis.append(func(val, *args, **kwargs))
                except : 
                    try : lis.append(listval.func(val, *args, **kwargs))
                    except : 
                        try : lis.append(func(listval, val, *args, **kwargs))
                        except : raise IlistError("unable to apply func")        
        if len(lis) == 1 : return lis[0]
        return lis

    @staticmethod
    def _index(idx, val): return idx.index(val)

    def _idxfull(self, minind=True):
        if minind : 
            iidxr = [set(self.iidx[i]) for i in range(self.lenidx) if not self.idxcoupled[i]]
            nidx = self._transpose(list(product(*iidxr)))
            for idx in range(self.lenidx): 
                if self.idxcoupled[idx]: 
                    dic = self._idxlink(self.iidx[self.idxref[idx]], self.iidx[idx])
                    nidx.insert(idx, [dic[i] for i in nidx[self.idxref[idx]]])
        else :      
            iidxr = [set(self.iidx[i]) for i in range(self.lenidx)]
            nidx = self._transpose(list(product(*iidxr)))
        return nidx

    @staticmethod
    def _idxlink(ref, l2):
        lis = set(Ilist._tuple(Ilist._transpose([ref, l2])))
        if not (len(lis) == len(set(ref)) == len(set(l2))) : return {}
        return dict(lis)    
    
    @staticmethod
    def _derived(ref, l2):
        # ref is derived from l2
        lis = set(Ilist._tuple(Ilist._transpose([ref, l2])))
        return len(lis) == len(set(l2)) and len(set(ref)) < len(set(l2))
    

    @staticmethod
    def _json(val): 
        if type(val) in [str, int, float, bool, tuple, list] : return val
        else : 
            try     : return val.json(json_string=False)
            except  : return val.to_json(json_string=False)

    @staticmethod
    def _list(idx): return list(map(list, idx))

    @staticmethod
    def _mul(val):
        mul = 1
        for v in val : mul *= v
        return mul
    
    @staticmethod
    def _nullValue(Class, fillvalue):
        if   type(fillvalue) == Class   : return fillvalue
        elif Class == int               : return 0
        elif Class == float             : return float("nan")
        elif Class == str               : return '-'
        elif Class == bool              : return False
        else                            : return Class()

    @staticmethod
    def _reindexidx(iidx, setidx, newsetidx) :
        return [newsetidx.index(setidx[indb]) for indb in iidx]
    
    @staticmethod
    def _reorder(val, idx=[]): 
        if idx == [] : return val 
        else : return [val[ind] for ind in idx]
        
    @staticmethod
    def _setable(extv):
        try : 
            set(extv)
            return extv
        except :
            return Ilist._tuple(extv)

    @staticmethod
    def _toint(extv, extset):
        ext = Ilist._setable(extv)
        return [extset.index(val) for val in ext]

    #def _toival(self, extval): return self._toset(self.extval).index(extval)
    def _toival(self, extval): return self.setval.index(extval)

    @staticmethod    
    def _toext(iidx, extset):
        return [extset[idx] for idx in iidx]

    @staticmethod    
    def _tonumpy(lis, func=identity, **kwargs):
        if func == None : func = identity
        if func == 'index' : return np.array(list(range(len(lis))))
        valList = Ilist._funclist(lis, func, **kwargs)
        if type(valList[0]) == str :
            try : datetime.fromisoformat(valList[0])
            except : return np.array(valList)
            return np.array(valList, dtype=np.datetime64)
        elif type(valList[0]) == datetime : return np.array(valList, dtype=np.datetime64)
        else: return np.array(valList)

    @staticmethod
    def _toset(extv):
        ext = Ilist._setable(extv)
        return list(set(ext))
        #return sorted(list(set(ext)))
            
    @staticmethod
    def _transpose(idx): 
        #if type(idx) != list or type(idx[0]) != list: raise IlistError('index not transposable')
        if type(idx) != list : raise IlistError('index not transposable')
        elif idx == [] : return []
        else : return [[ix[ind] for ix in idx] for ind in range(len(idx[0]))]

    @staticmethod
    def _tuple(idx): return list(map(tuple, idx))
    
    def _updateset(self, extidx):
        # ajouter à la fin un recalcul de iidx en fonction de sorted(extset)
        iidx = []
        for i in range(len(extidx)):
            if len(self.setidx) == i : self.setidx.append([])
            if extidx[i] not in self.setidx[i] : self.setidx[i].append(extidx[i])
            iidx.append(self.setidx[i].index(extidx[i]))
        return iidx
            
    def _xcoord(self, funcidx=[], **kwargs) :
        ''' Coords generation for Xarray'''
        coord = {}
        for i in range(self.lenidx):
            if funcidx==[] : funci=identity 
            else : funci= funcidx[i]
            xlisti = self._tonumpy(self.setidx[i], func=funci, **kwargs)
            if not self.idxcoupled[i] : coord[self.idxname[i]] = xlisti
            else : coord[self.idxname[i]] = (self.idxname[self.idxref[i]], xlisti)
        return coord


class IlistError(Exception):
    pass
    

    '''def to_numpy(self, func=None, ind='axe', squeeze=True, fillvalue='?', **kwargs):
        # ind : axe, all, flat
        if not self.consistent : raise IlistError("unable to generate numpy array with inconsistent Ilist")
        if   type(ind) == str and ind == 'flat' : 
            return self._to_numpy(self.extval, func=func, **kwargs)
            if func == 'index' : return np.array(self.ival)
            elif dtype == None : return np.array(self._funclist(self.extval, func, **kwargs))
            else : return np.array(self._funclist(self.extval, func, **kwargs), dtype=dtype)'''
    '''elif type(ind) == str and ind in ['axe', 'all']  :
            ilf = self.full(minind=ind=='axe', fillvalue=fillvalue)
            ilf.sort(order=self.axes)
            return self._to_numpy(ilf.extval, func=func, **kwargs).reshape(ilf.axeslen)'''
    '''if func == 'index' : return np.array(ilf.ival)                                  .reshape(ilf.axeslen)
            elif dtype == None : return np.array(self._funclist(ilf.extval, func, **kwargs)).reshape(ilf.axeslen)
            else : return np.array(self._funclist(ilf.extval, func, **kwargs), dtype=dtype) .reshape(ilf.axeslen)'''
        
    '''if type(self.extval[0]) == str :
            try : 
                datetime.fromisoformat(self.extval[0])
                dtype = np.datetime64
            except : dtype=None
        elif type(self.extval[0]) == datetime : dtype = np.datetime64
        else : dtype = None
        if func == None : func = identity
        if   type(ind) == str and ind == 'flat' : 
            if func == 'index' : return np.array(self.ival)
            elif dtype == None : return np.array(self._funclist(self.extval, func, **kwargs))
            else : return np.array(self._funclist(self.extval, func, **kwargs), dtype=dtype)
        elif type(ind) == str and ind in ['axe', 'all']  :
            ilf = self.full(minind=ind=='axe', fillvalue=fillvalue)
            ilf.sort(order=self.axes)
            if func == 'index' : return np.array(ilf.ival)                                  .reshape(ilf.axeslen)
            elif dtype == None : return np.array(self._funclist(ilf.extval, func, **kwargs)).reshape(ilf.axeslen)
            else : return np.array(self._funclist(ilf.extval, func, **kwargs), dtype=dtype) .reshape(ilf.axeslen)'''


