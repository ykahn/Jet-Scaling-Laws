// gen_Zqq.cc
// Particle gun: two back-to-back d/dbar quarks at E = mZ/2 each.
// QCD FSR on, QED off, hadronization off.
//
// Writes 100000 events to ../data/Zqq_events.txt
// Format: one line per final-state particle (E px py pz), blank line between events.

#include "Pythia8/Pythia.h"
#include <fstream>
#include <string> 
using namespace Pythia8;

int main(int argc, char* argv[]) {
    const double mZ    = 91.188;
    const double Equark = mZ / 2.0;

    Pythia pythia;
// Skip hard scatter generation
    pythia.readString("ProcessLevel:all = off");
    pythia.settings.parm("Beams:eCM", mZ);

    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:MPI = off");
    pythia.readString("PartonLevel:FSR = on");
// No QED in FSR
    pythia.readString("TimeShower:QEDshowerByQ = off");
    pythia.readString("TimeShower:QEDshowerByL = off");
    pythia.readString("TimeShower:QEDshowerByGamma = off");
// Stop at parton level: no hadronization, no hadron decays.
    pythia.readString("HadronLevel:all = off");
// fix alpha 
    double alphaS = (argc > 1) ? atof(argv[1]) : 0.1365;           // ./gen_Zqq 0.12
    int    order  = (argc > 2) ? atoi(argv[2]) : 1;             // 0 = fixed, 1 = one-loop running (Pythia default)
    pythia.settings.mode("TimeShower:alphaSorder", order);      // mode() for ints, not parm(); delete the readString line above it
    pythia.settings.parm("TimeShower:alphaSvalue", alphaS);

    pythia.readString("Next:numberCount = 1000");
    pythia.init();

// open .txt file 
    std::string tag = (argc > 1 ? std::string(argv[1]) : "0.1365") + (order == 0 ? "_fixed" : "_run");
    std::string fname = "../data/Zqq_events_as" + tag + ".txt";
    std::ofstream out(fname);
    const int nEvents = 100000;
    int nWritten = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        pythia.event.reset();
// initial state
        pythia.event.append( 1, 23, 0, 0, 0, 0, 101,   0,  0., 0.,  Equark, Equark, 0.);
        pythia.event.append(-1, 23, 0, 0, 0, 0,   0, 101,  0., 0., -Equark, Equark, 0.);
        pythia.event[1].scale(Equark);
        pythia.event[2].scale(Equark);
        pythia.event.scale(Equark);
// run FSR shower explicitly on particles 1 and 2
        pythia.forceTimeShower(1, 2, Equark);
        if (!pythia.next()) continue;
        if (iEvent == 0) pythia.event.list();
// write final state info to file, skipping neutrinos
        for (int i = 0; i < pythia.event.size(); ++i) {
            Particle& p = pythia.event[i];
            if (!p.isFinal()) continue;
            int aid = abs(p.id());
            if (aid == 12 || aid == 14 || aid == 16) continue;
            out << p.e()  << " " << p.px() << " "
                << p.py() << " " << p.pz() << "\n";
        }
        out << "\n";
        ++nWritten;
    }

    out.close();
    std::cout << "Wrote " << nWritten << " events to " + fname;
    pythia.stat();
    return 0;
}
