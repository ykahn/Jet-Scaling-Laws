// gen_Hgg.cc
// Particle gun: two back-to-back gluons at E = mZ/2 each (mH = mZ fiction).
// QCD FSR on, QED off, hadronization off.
//
// Writes 100000 events to ../data/Hgg_events.txt
// Format: one line per final-state particle (E px py pz), blank line between events.

#include "Pythia8/Pythia.h"
#include <fstream>
using namespace Pythia8;

int main() {
    const double mZ    = 91.188;
    const double Eglue = mZ / 2.0;

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

    pythia.readString("Next:numberCount = 1000");

    pythia.init();

    std::ofstream out("../data/Hgg_events.txt");
    const int nEvents = 100000;
    int nWritten = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        pythia.event.reset();

        pythia.event.append(21, 23, 0, 0, 0, 0, 101, 102,  0., 0.,  Eglue, Eglue, 0.);
        pythia.event.append(21, 23, 0, 0, 0, 0, 102, 101,  0., 0., -Eglue, Eglue, 0.);
        pythia.event[1].scale(Eglue);
        pythia.event[2].scale(Eglue);
        pythia.event.scale(Eglue);
// run FSR shower explicitly on particles 1 and 2
        pythia.forceTimeShower(1, 2, Eglue);
        if (!pythia.next()) continue;

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
    std::cout << "Wrote " << nWritten << " events to ../data/Hgg_events.txt\n";
    pythia.stat();
    return 0;
}
