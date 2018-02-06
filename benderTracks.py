from Bender.Main import *
from Bender.MainMC import *
import math
import numpy as np
from ROOT import Double, TLorentzVector, TVector3, TRotation, TLorentzRotation, TMath

class TagInfo(AlgoMC):

    def __init__( self, name, isMC = True, year = 2012, **kwargs ):
        super(TagInfo, self).__init__( name, **kwargs )

        self.isMC = isMC

        self.year = year

        self.setupFunctors()

        self.bkgtools = []
        if self.isMC :
            # the first of these can't be used in Bender <= v18r3
            self.bkgtools.append( self.tool( cpp.IBackgroundCategory, 'BackgroundCategoryViaRelations' ) )
            self.bkgtools.append( self.tool( cpp.IBackgroundCategory, 'BackgroundCategory' ) )

        self.tistostoolL0   = self.tool( cpp.ITriggerTisTos, 'L0TriggerTisTos' )
        self.tistostoolHLT1 = self.tool( cpp.ITriggerTisTos, 'Hlt1TriggerTisTos' )
        self.tistostoolHLT2 = self.tool( cpp.ITriggerTisTos, 'Hlt2TriggerTisTos' )


        # HLT 1
        self.hlt1triggerlist = {}

        self.hlt1triggerlist[2011] = ['Hlt1TrackAllL0Decision',
                                      'Hlt1TrackMuonDecision',
                                      'Hlt1TrackPhotonDecision',
                                      'Hlt1DiMuonHighMassDecision',
                                      'Hlt1TrackMuonDecision',
                                     ]

        self.hlt1triggerlist[2012] = self.hlt1triggerlist[2011]

        self.hlt1triggerlist[2015] = ['Hlt1TrackMVADecision',
                                      'Hlt1TwoTrackMVADecision',
                                      'Hlt1TrackMuonDecision',
                                      'Htl1IncPhiDecision',
                                      'Hlt1TrackMVADecision',
                                      'Hlt1DiMuonHighMassDecision']

        self.hlt1triggerlist[2016] = self.hlt1triggerlist[2015][:-1] + ['Hlt1TrackMVALooseDecision',
                                                                        'Hlt1TwoTrackMVALooseDecision',
                                                                        'Hlt1PhiIncPhiDecision']

        self.hlt1triggerlist[2017] = self.hlt1triggerlist[2016]


        # HLT 2
        self.hlt2triggerlist = {}

        self.hlt2triggerlist[2011] = ['Hlt2Topo2BodyBBDTDecision',
                                      'Hlt2Topo3BodyBBDTDecision',
                                      'Hlt2Topo4BodyBBDTDecision',
                                      'Hlt2Topo2BodySimpleDecision',
                                      'Hlt2Topo3BodySimpleDecision',
                                      'Hlt2Topo4BodySimpleDecision',
                                      'Hlt2B2HHDecision',
                                      'Hlt2B2HHPi0_MergedDecision',
                                      'Hlt2DiMuonJPsiDecision',
                                      'Hlt2DiMuonDetachedJPsiDecision']

        self.hlt2triggerlist[2012] = self.hlt2triggerlist[2011]

        self.hlt2triggerlist[2015] = ['Hlt2IncPhiDecision',
                                      'Hlt2Topo2BodyDecision',
                                      'Hlt2Topo3BodyDecision',
                                      'Hlt2Topo4BodyDecision',
                                      'Hlt2B2Kpi0',
                                      'Hlt2B2K0pi0',
                                      'Hlt2DiMuonJPsiDecision',
                                      'Hlt2DiMuonDetachedJPsiDecision']

        self.hlt2triggerlist[2016] = ['Hlt2PhiIncPhiDecision',
                                      'Hlt2Topo2BodyDecision',
                                      'Hlt2Topo3BodyDecision',
                                      'Hlt2Topo4BodyDecision',
                                      'Hlt2B2Kpi0_B2K0pi0',
                                      'Hlt2B2Kpi0_B2Kpi0',
                                      'Hlt2DiMuonJPsiDecision',
                                      'Hlt2DiMuonDetachedJPsiDecision']

        self.hlt2triggerlist[2017] = self.hlt2triggerlist[2016]

    def fillSignalVertices(self, bCand):

        signalVertices = []

        signalVertices.append(bCand.endVertex())

        # Just consider 2b decays for now
        for c in bCand.children():
            if c.endVertex():
                signalVertices.append(c.endVertex())

        return signalVertices

    def fillSignalParticles(self, tracks, p):

        if p.proto():
            tracks.append(p.proto())
            return
        else:
            for c in p.children():
                self.fillSignalParticles(tracks, c)

    def unique_cands( self, cands ) :
        """
        Find unique candidates in a list
        """

        unique_cands = []
        evt_keys = set()
        for cand in cands :

            cand_keys = set()
            for daug in cand.children() :
                cand_keys.add( daug.key() )

            if cand_keys in evt_keys :
                continue
            else :
                unique_cands.append( cand )
                evt_keys.add( frozenset(cand_keys) )

        return unique_cands


    def get_ntracks( self ) :
        """
        Extracts the number of Best and Long tracks in the event
        """

        nbest = 0
        nlong = 0
        nspdhits = 0
        nrich1hits = 0
        nrich2hits = 0

        try :
            summary = self.get('/Event/Rec/Summary')
            nbest = summary.info(summary.nTracks,0)
            nlong = summary.info(summary.nLongTracks,0)
            nspdhits = summary.info(summary.nSPDhits,0)
            nrich1hits = summary.info(summary.nRich1Hits,0)
            nrich2hits = summary.info(summary.nRich2Hits,0)
        except :
            try :
                tracks = self.get( 'Rec/Track/Best' )
                nlong = 0
                nbest = len(tracks)
                for trk in tracks :
                    if 3 == trk.type() :
                        nlong += 1
            except :
                self.Error( 'Information about number of tracks not found neither in /Event/Rec/Summary nor in Rec/Track/Best', SUCCESS )

        return nbest, nlong, nspdhits, nrich1hits, nrich2hits

    def setupFunctors(self):

        # on protoparticle

        self.velocharge = PPINFO(LHCb.ProtoParticle.VeloCharge, -99)
        self.probnnghost = PPINFO(LHCb.ProtoParticle.ProbNNghost, -99)
        self.isnote = PPINFO(LHCb.ProtoParticle.IsNotE, -99)
        self.isnoth = PPINFO(LHCb.ProtoParticle.IsNotH, -99)
        self.isphoton = PPINFO(LHCb.ProtoParticle.IsPhoton, -99)
        self.calotrmatch = PPINFO(LHCb.ProtoParticle.CaloTrMatch, -99)

    def bTruth(self, theTuple, bCand):

        if not bCand.empty() :

            bCand = bCand[0]

            theTuple.column_int( 'B_TRUEID', int(MCID(bCand)) )

            theTuple.column_int( 'B_TRUEQ', int(MC3Q(bCand)/3) )

            theTuple.column_double( 'B_TRUEP', MCP(bCand) )
            theTuple.column_double( 'B_TRUEPE', MCE(bCand) )
            theTuple.column_double( 'B_TRUEPX', MCPX(bCand) )
            theTuple.column_double( 'B_TRUEPY', MCPY(bCand) )
            theTuple.column_double( 'B_TRUEPZ', MCPZ(bCand) )

        else:

            theTuple.column_int( 'B_TRUEID', -99 )

            theTuple.column_int( 'B_TRUEQ', -99 )

            theTuple.column_double( 'B_TRUEP', -99. )
            theTuple.column_double( 'B_TRUEPE', -99. )
            theTuple.column_double( 'B_TRUEPX', -99. )
            theTuple.column_double( 'B_TRUEPY', -99. )
            theTuple.column_double( 'B_TRUEPZ', -99. )

    def bkg_category_tuple( self, theTuple, cand) :
        """
        Reproduce information stored by TupleToolMCBackgroundInfo
        """

        category = -1
        for tool in self.bkgtools :
            category = tool.category( cand )
            if category > -1 and category < 1001 :
                break

        if category > 1000 :
            category = -1

        theTuple.column_int( 'B_BKGCAT', category )

    def trig_tuple( self, theTuple, bcand, particleName = '', storeHLT2 = True ) :

        # For S24 at least, "ERROR HltSelReportsDecoder:: Failed to add Hlt selection name Hlt2RecSummary to its container'
        # can be ignored

        hdrL0 = self.get('/Event/HltLikeL0/DecReports')
        tckL0 = -1
        if hdrL0 :
            tckL0 = hdrL0.configuredTCK()
        theTuple.column_int("tckL0", tckL0 )

        hdrHLT1 = self.get('/Event/Hlt1/DecReports')
        tckHLT1 = -1
        if hdrHLT1 :
            tckHLT1 = hdrHLT1.configuredTCK()
        theTuple.column_int("tckHLT1", tckHLT1 )

        hdrHLT2 = self.get('/Event/Hlt2/DecReports')
        tckHLT2 = -1
        if hdrHLT2 :
            tckHLT2 = hdrHLT2.configuredTCK()
        theTuple.column_int("tckHLT2", tckHLT2 )

        # Setup the TISTOS tools
        self.tistostoolL0.setOfflineInput( bcand )
        self.tistostoolHLT1.setOfflineInput( bcand )
        self.tistostoolHLT2.setOfflineInput( bcand )

        # Get the L0 decisions, plus TIS & TOS info

        l0triggers = self.tistostoolL0.triggerSelectionNames('L0.*Decision')
        self.tistostoolL0.setTriggerInput( 'L0.*Decision' )
        l0 = self.tistostoolL0.tisTosTobTrigger()
        l0dec = l0.decision()
        l0tis = l0.tis()
        l0tos = l0.tos()
        theTuple.column_int((particleName + "_" if particleName else "") +  "L0Global_Dec", l0dec)
        theTuple.column_int((particleName + "_" if particleName else "") +  "L0Global_TIS", l0tis)
        theTuple.column_int((particleName + "_" if particleName else "") +  "L0Global_TOS", l0tos)

        l0declist = self.tistostoolL0.triggerSelectionNames( self.tistostoolL0.kTrueRequired, self.tistostoolL0.kAnything,     self.tistostoolL0.kAnything     )
        l0tislist = self.tistostoolL0.triggerSelectionNames( self.tistostoolL0.kTrueRequired, self.tistostoolL0.kTrueRequired, self.tistostoolL0.kAnything     )
        l0toslist = self.tistostoolL0.triggerSelectionNames( self.tistostoolL0.kTrueRequired, self.tistostoolL0.kAnything,     self.tistostoolL0.kTrueRequired )

        # set the list of L0 decisions to be stored
        l0tuplelist = ['L0DiMuonDecision',
                       'L0MuonDecision',
                       'L0ElectronDecision',
                       'L0ElectronHiDecision',
                       'L0PhotonDecision',
                       'L0HadronDecision']

        for line in l0triggers :
            if line in l0tuplelist :
                l0dec = 0
                l0tis = 0
                l0tos = 0
                if line in l0declist : l0dec = 1
                if line in l0tislist : l0tis = 1
                if line in l0toslist : l0tos = 1
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_Dec", l0dec)
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_TIS", l0tis)
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_TOS", l0tos)

        # Get the HLT decisions, plus TIS & TOS info

        hlt1triggers = self.tistostoolHLT1.triggerSelectionNames('Hlt1.*Decision')
        self.tistostoolHLT1.setTriggerInput( 'Hlt1.*Decision' )
        hlt1 = self.tistostoolHLT1.tisTosTobTrigger()
        hlt1dec = hlt1.decision()
        hlt1tis = hlt1.tis()
        hlt1tos = hlt1.tos()
        theTuple.column_int((particleName + "_" if particleName else "") + "Hlt1Global_Dec", hlt1dec)
        theTuple.column_int((particleName + "_" if particleName else "") + "Hlt1Global_TIS", hlt1tis)
        theTuple.column_int((particleName + "_" if particleName else "") + "Hlt1Global_TOS", hlt1tos)

        hlt1declist = self.tistostoolHLT1.triggerSelectionNames( self.tistostoolHLT1.kTrueRequired, self.tistostoolHLT1.kAnything,     self.tistostoolHLT1.kAnything     )
        hlt1tislist = self.tistostoolHLT1.triggerSelectionNames( self.tistostoolHLT1.kTrueRequired, self.tistostoolHLT1.kTrueRequired, self.tistostoolHLT1.kAnything     )
        hlt1toslist = self.tistostoolHLT1.triggerSelectionNames( self.tistostoolHLT1.kTrueRequired, self.tistostoolHLT1.kAnything,     self.tistostoolHLT1.kTrueRequired )

        for line in self.hlt1triggerlist[self.year] :
            hlt1dec = 0
            hlt1tis = 0
            hlt1tos = 0
            if line in hlt1declist : hlt1dec = 1
            if line in hlt1tislist : hlt1tis = 1
            if line in hlt1toslist : hlt1tos = 1
            theTuple.column_int((particleName + "_" if particleName else "") + line+"_Dec", hlt1dec)
            theTuple.column_int((particleName + "_" if particleName else "") + line+"_TIS", hlt1tis)
            theTuple.column_int((particleName + "_" if particleName else "") + line+"_TOS", hlt1tos)

        if storeHLT2:

            hlt2triggers = self.tistostoolHLT2.triggerSelectionNames('Hlt2.*Decision')
            self.tistostoolHLT2.setTriggerInput( 'Hlt2.*Decision' )
            hlt2 = self.tistostoolHLT2.tisTosTobTrigger()
            hlt2dec = hlt2.decision()
            hlt2tis = hlt2.tis()
            hlt2tos = hlt2.tos()
            theTuple.column_int((particleName + "_" if particleName else "") + "Hlt2Global_Dec", hlt2dec)
            theTuple.column_int((particleName + "_" if particleName else "") + "Hlt2Global_TIS", hlt2tis)
            theTuple.column_int((particleName + "_" if particleName else "") + "Hlt2Global_TOS", hlt2tos)

            hlt2declist = self.tistostoolHLT2.triggerSelectionNames( self.tistostoolHLT2.kTrueRequired, self.tistostoolHLT2.kAnything,     self.tistostoolHLT2.kAnything     )
            hlt2tislist = self.tistostoolHLT2.triggerSelectionNames( self.tistostoolHLT2.kTrueRequired, self.tistostoolHLT2.kTrueRequired, self.tistostoolHLT2.kAnything     )
            hlt2toslist = self.tistostoolHLT2.triggerSelectionNames( self.tistostoolHLT2.kTrueRequired, self.tistostoolHLT2.kAnything,     self.tistostoolHLT2.kTrueRequired )

            for line in self.hlt2triggerlist[self.year] :
                hlt2dec = 0
                hlt2tis = 0
                hlt2tos = 0
                if line in hlt2declist : hlt2dec = 1
                if line in hlt2tislist : hlt2tis = 1
                if line in hlt2toslist : hlt2tos = 1
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_Dec", hlt2dec)
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_TIS", hlt2tis)
                theTuple.column_int((particleName + "_" if particleName else "") + line+"_TOS", hlt2tos)



    def trackInfo(self, theTuple, track):

        bestPV = self.bestVertex( track )

        mass = M(track)

        p = track.p()
        pt = track.pt()
        charge = track.charge()

        eta = ETA(track)
        phi = PHI(track)

        hasRICH = HASRICH(track)

        ctau = CTAU(bestPV)(track)

        px = PX(track)
        py = PY(track)
        pz = PZ(track)
        e = E(track)

        trchi2 = TRCHI2(track)
        trchi2ndof = TRCHI2DOF(track)
        trghostprob = TRGHOSTPROB(track)

        pide = PIDe(track)
        pidmu = PIDmu(track)
        pidk = PIDK(track)
        pidpi = PIDpi(track)
        pidp = PIDp(track)

        ismuon = ISMUON(track)
        isdown = ISDOWN(track)
        islong = ISLONG(track)

        # on protoparticle

        velocharge = self.velocharge(track)
        probnnghost = self.probnnghost(track)
        isnote = self.isnote(track)
        isnoth = self.isnoth(track)
        isphoton = self.isphoton(track)
        calotrmatch = self.calotrmatch(track)

        theTuple.column_double( 'mass', mass )
        theTuple.column_double( 'P', p )
        theTuple.column_double( 'PX', px)
        theTuple.column_double( 'PY', py )
        theTuple.column_double( 'PZ', pz )
        theTuple.column_double( 'PT', pt )
        theTuple.column_double( 'E', e )
        theTuple.column_double( 'charge', charge )
        theTuple.column_double( 'eta', eta )
        theTuple.column_double( 'phi', phi )
        theTuple.column_double( 'hasrich', hasRICH )
        theTuple.column_double( 'ctau', ctau )
        theTuple.column_double( 'trchi2', trchi2 )
        theTuple.column_double( 'trchi2ndof', trchi2ndof )
        theTuple.column_double( 'trghostprob', trghostprob )
        theTuple.column_double( 'pide', pide )
        theTuple.column_double( 'pidmu', pidmu )
        theTuple.column_double( 'pidk', pidk )
        theTuple.column_double( 'pidpi', pidpi )
        theTuple.column_double( 'pidp', pidp )
        theTuple.column_double( 'ismuon', ismuon )
        theTuple.column_double( 'islong', islong )
        theTuple.column_double( 'isdown', isdown )
        theTuple.column_double( 'velocharge', velocharge )
        theTuple.column_double( 'probnnghost', probnnghost )
        theTuple.column_double( 'isnote', isnote )
        theTuple.column_double( 'isphoton', isphoton )
        theTuple.column_double( 'calotrmatch', calotrmatch )

    def vertexInfo(self, theTuple, particle): # operate on the mother particle as vertex

        # Also add in IP info for particle PV and B PV

        vx = VFASPF(VX)(particle)
        vy = VFASPF(VY)(particle)
        vz = VFASPF(VZ)(particle)

        vtxchi2ndof = VFASPF(VCHI2)(particle) / VFASPF(VDOF)(particle)

        distPrimary = VFASFP(VMINVDDV(PRIMARY))(particle)

        p = particle.p()
        pt = particle.pt()
        charge = particle.charge()

        eta = ETA(particle)
        phi = PHI(particle)

        mass = M(particle)

        daugs = particle.daughters()

        nDaugs = len(daugs)


    def ip_tuple( self, theTuple, particle, name, bCand = None ) :
        """
        Store the impact parameter info for the particle
        """
        primaries = self.vselect('PV', PRIMARY )
        bestPV = self.bestVertex( particle )

        if bCand:
            bestPV = self.bestVertex( bCand )

        minipfun     = MINIP( primaries, self.geo() )
        minipchi2fun = MINIPCHI2( primaries, self.geo() )
        ipbpvfun     = IP( bestPV, self.geo() )
        ipchi2bpvfun = IPCHI2( bestPV, self.geo() )

        theTuple.column_double( name + '_MINIP',         minipfun(particle)     )
        theTuple.column_double( name + '_MINIPCHI2',     minipchi2fun(particle) )
        theTuple.column_double( name + '_IP_OWNPV',      ipbpvfun(particle)     )
        theTuple.column_double( name + '_IPCHI2_OWNPV',  ipchi2bpvfun(particle) )

        if not bCand:

            bpvpos = bestPV.position()
            theTuple.column_double( name + '_OWNPV_X', bpvpos.x() )
            theTuple.column_double( name + '_OWNPV_Y', bpvpos.y() )
            theTuple.column_double( name + '_OWNPV_Z', bpvpos.z() )

            covMatrix = bestPV.covMatrix()
            theTuple.column_double( name + '_OWNPV_XERR', TMath.Sqrt( covMatrix(0,0) ) )
            theTuple.column_double( name + '_OWNPV_YERR', TMath.Sqrt( covMatrix(1,1) ) )
            theTuple.column_double( name + '_OWNPV_ZERR', TMath.Sqrt( covMatrix(2,2) ) )

            chi2 = bestPV.chi2()
            ndof = bestPV.nDoF()
            chi2ndof = bestPV.chi2PerDoF()
            theTuple.column_int( name + '_OWNPV_NDOF',     ndof                     )
            theTuple.column_double( name + '_OWNPV_CHI2',     chi2                     )
            theTuple.column_double( name + '_OWNPV_CHI2NDOF', chi2ndof                 )
            theTuple.column_double( name + '_OWNPV_PROB',     TMath.Prob( chi2, ndof ) )

            ntrk = bestPV.tracks().size()
            theTuple.column_int( name + '_OWNPV_NTRACKS', ntrk )


    def vtx_tuple( self, theTuple, name, particle, bCand = None ) :
        """
        Store vertex info for the Xib
        """

        vertex = particle.endVertex()

        vtxpos = vertex.position()
        theTuple.column_double( name + '_ENDVERTEX_X',    vtxpos.x() )
        theTuple.column_double( name + '_ENDVERTEX_Y',    vtxpos.y() )
        theTuple.column_double( name + '_ENDVERTEX_Z',    vtxpos.z() )

        covMatrix = vertex.covMatrix()
        theTuple.column_double( name + '_ENDVERTEX_XERR', TMath.Sqrt( covMatrix(0,0) ) )
        theTuple.column_double( name + '_ENDVERTEX_YERR', TMath.Sqrt( covMatrix(1,1) ) )
        theTuple.column_double( name + '_ENDVERTEX_ZERR', TMath.Sqrt( covMatrix(2,2) ) )

        chi2 = vertex.chi2()
        ndof = vertex.nDoF()
        chi2ndof = vertex.chi2PerDoF()
        theTuple.column_int( name + '_ENDVERTEX_NDOF',     ndof                     )
        theTuple.column_double( name + '_ENDVERTEX_CHI2',     chi2                     )
        theTuple.column_double( name + '_ENDVERTEX_CHI2NDOF', chi2ndof                 )
        theTuple.column_double( name + '_ENDVERTEX_PROB',     TMath.Prob( chi2, ndof ) )

        primaries = self.vselect('PV', PRIMARY )
        minvdfun     = MINVVD( primaries )
        minvdchi2fun = MINVVDCHI2( primaries )

        theTuple.column_double( name + '_MINVD',     minvdfun(particle)     )
        theTuple.column_double( name + '_MINVDCHI2', minvdchi2fun(particle) )

        bestPV = self.bestVertex( particle )
        if bCand:
            bestPV = self.bestVertex( bCand )
        vdfun     = VD( bestPV )
        vdchi2fun = VDCHI2( bestPV )
        dirafun   = DIRA( bestPV )

        theTuple.column_double( name + '_VD_OWNPV',     vdfun(particle)     )
        theTuple.column_double( name + '_VDCHI2_OWNPV', vdchi2fun(particle) )
        theTuple.column_double( name + '_DIRA_OWNPV',   dirafun(particle)   )

    def updateVertexList(self, currentVertices, newVertices):

        # Track can be shared by two vertices, one of which is better, one of which is worse

        verticesToAdd = set()

        for v in newVertices:
            thisChi2 = v.endVertex().chi2PerDoF()
            thisKids = v.children()
            newVertices = []

            for otherV, otherChi2 in currentVertices:
                otherKids = [otherV.children()[0].proto(), otherV.children()[1].proto()]

                if thisKids[0].proto() in otherKids or thisKids[1].proto() in otherKids:
                    # If there's an overlap in tracks
                    # Take the best vertex

                    if thisChi2 > otherChi2:
                        newVertices.append((otherV, otherChi2))

                        if (v, thisChi2) in verticesToAdd:
                            # This was better than another vertex, but is now worse than this one
                            # so remove it

                            # If the vertices in currentVertices don't overlap with tracks, this should never get called
                            verticesToAdd.remove((v, thisChi2))
                    else:
                        # New vertex can overlap with multiple, but just need one instance - use Python set
                        verticesToAdd.add((v, thisChi2))

                else:
                    # Otherwise take both
                    newVertices.append((otherV, otherChi2))
                    verticesToAdd.add((v, thisChi2))

            # Update current vertices with those removed that overlap
            currentVertices = newVertices

        currentVertices.extend(list(verticesToAdd))

        return currentVertices

    def analyse( self ) :
        """
        The main 'analysis' method
        """

        bTuple = self.nTuple( 'B' )
        trackTuple = self.nTuple( 'tracks' )
        vertexTuple = self.nTuple( 'vertices' )

        vertices = []

        pionVtxs = get( '/Event/Phys/StdLooseDetachedDipion/Particles' ) # 2mm flight distance! Rho?
        muonVtxs = get( '/Event/Phys/StdLooseDiMuon/Particles' )
        d0Vtxs = get( '/Event/Phys/StdLooseD02KPi/Particles' )

        # Greedy - doesn't solve the assignment problem (i.e., doesn't guarantee globally minimised total vtx chisq)

        for v in pionVtxs:
            chi2ndof = v.endVertex().chi2PerDoF()
            vertices.append((v, chi2ndof))

        vertices = self.updateVertexList(vertices, muonVtxs)
        vertices = self.updateVertexList(vertices, d0Vtxs)

        cands = self.select( 'candidates', '[B+ -> (J/psi(1S) -> mu+ mu-) K+]CC' )
        if not cands:
            return SUCCESS
        cands = self.unique_cands( cands )
        nCands = len(cands)

        pions = get( '/Event/Phys/StdNoPIDsPions/Particles' )
        electrons = get( '/Event/Phys/StdAllLooseElectrons/Particles' )
        muons = get( '/Event/Phys/StdAllLooseMuons/Particles' )

        tracks = []

        for p in pions:
            tracks.append(p)

        for p in electrons:
            if not any([p.proto() == x.proto() for x in tracks]):
                tracks.append(p)
        for p in muons:
            if not any([p.proto() == x.proto() for x in tracks]):
                tracks.append(p)

        bCand = cands[0] # Just take the first, for the moment

        signalParticles = []
        self.fillSignalParticles(signalParticles, bCand)

        evthdr = self.get( '/Event/Rec/Header' )
        runNum = evthdr.runNumber()
        evtNum = evthdr.evtNumber()

        nbest, nlong, nspdhits, nrich1hits, nrich2hits = self.get_ntracks()

        bTuple.column_int( 'nTracks', nbest )
        bTuple.column_int( 'nLong', nlong )
        bTuple.column_int( 'nSPDHist', nspdhits )
        bTuple.column_int( 'nRICH1Hits', nrich1hits )
        bTuple.column_int( 'nRICH2Hits', nrich2hits )

        bTuple.column_int( 'runNumber', runNum )
        bTuple.column_int( 'evtNumber', evtNum )

        self.vtx_tuple(bTuple, 'B', bCand)
        self.ip_tuple(bTuple, bCand, 'B' )
        self.trackInfo(bTuple, bCand)

        tracksInVertices = set()
        for v, chisq in vertices:
            tracksInVertices.add(v.children()[0].proto())
            tracksInVertices.add(v.children()[1].proto())

        nSelectedTracks = 0
        for iTrack, track in enumerate(tracks):

            if track.proto() in signalParticles:
                continue
            else:
                nSelectedTracks += 1

            self.ip_tuple(trackTuple, track, 'track' )
            self.ip_tuple(trackTuple, track, 'track_BPV', bCand = bCand )

            self.trackInfo(trackTuple, track)

            trackTuple.column_int( 'iTrack', iTrack )
            trackTuple.column_int( 'runNumber', runNum )
            trackTuple.column_int( 'evtNumber', evtNum )

            inVertex = -1

            try:
                inVertex = tracksInVertices.index(track.proto())
            except:
                inVertex = -1

            trackTuple.column_int('inVertex', inVertex)

            trackTuple.write()

        nSelectedVertices = 0
        for iVertex, vertex in enumerate(map(lambda x : x[0], vertices)):

            # Use signalParticles as these aren't RecVertex
            if vertex.children()[0].proto() in signalParticles or vertex.children()[1].proto() in signalParticles:
                continue
            else:
                nSelectedVertices += 1

            vertexTuple.column_int( 'iVertex', iVertex )
            vertexTuple.column_int( 'runNumber', runNum )
            vertexTuple.column_int( 'evtNumber', evtNum )

            self.ip_tuple(vertexTuple, vertex, 'vertex' )
            self.ip_tuple(vertexTuple, vertex, 'vertex_BPV', bCand = bCand )

            self.trackInfo(vertexTuple, vertex)

            self.vtx_tuple(vertexTuple, 'vertex', vertex)
            self.vtx_tuple(vertexTuple, 'B', vertex, bCand = bCand)

            vertexTuple.write()

        bTuple.column_int( 'nSelectedTracks', nSelectedTracks )
        bTuple.column_int( 'nSelectedVertices', nSelectedVertices )

        self.trig_tuple(bTuple, bCand, particleName = 'B')

        if self.isMC:
            mcB = self.mcselect( 'mcB', '[B+ -> (J/psi(1S) -> mu+ mu-) K+]CC' )

            self.bTruth(bTuple, mcB)
            self.bkg_category_tuple(bTuple, bCand)

        bTuple.write()

        self.setFilterPassed( True )

        return SUCCESS

def configure ( inputdata        ,
                catalogs = []    ,
                castor   = False ,
                params   = {}    ) :

    from Configurables import DaVinci

    daVinci = DaVinci ( DataType   = '2012' , InputType  = 'DST', Simulation = True )

    daVinci.TupleFile = 'B2JPsiK-TagInfo.root'

    setData  ( inputdata , catalogs , castor )  ## inform bender about input files and access to data

    from BenderTools.GetDBtags import getDBTags
    tags = getDBTags ( inputdata[0] , castor  )

    logger.info ( 'Extract tags from DATA : %s' % tags )
    if tags.has_key ( 'DDDB' ) and tags ['DDDB'] :
        daVinci.DDDBtag   = tags['DDDB'  ]
        logger.info ( 'Set DDDB    %s ' % daVinci.DDDBtag   )
    if tags.has_key ( 'CONDDB' ) and tags ['CONDDB'] :
        daVinci.CondDBtag = tags['CONDDB']
        logger.info ( 'Set CONDDB  %s ' % daVinci.CondDBtag )
    if tags.has_key ( 'SIMCOND' ) and tags ['SIMCOND'] :
        daVinci.CondDBtag = tags['SIMCOND']
        logger.info ( 'Set SIMCOND %s ' % daVinci.CondDBtag )

    gaudi = appMgr()

    alg = TagInfo('TagInfo', Inputs = ['/Event/AllStreams/Phys/BetaSBu2JpsiKDetachedLine/Particles', '/Event/AllStreams/Phys/BetaSBu2JpsiKPrescaledLine/Particles'])

    gaudi.setAlgorithms( [ alg ] )

    return SUCCESS

if __name__ == '__main__' :

    inputdata = [
        '/data/lhcb/phrnas/Bu_JPsiK_2012_MD_Sim08a_S20.dst'
        ]

    configure( inputdata , castor = False )

    run(100)

# =============================================================================
