// gen_Hgg.cc
// Particle gun: two back-to-back gluons at E = mZ/2 each, in a color-singlet
// configuration matching H(mH=mZ)->gg. Pythia showers and hadronizes them.
//
// Writes 10,000 events to ../data/Hgg_events.txt
// Format: one line per final-state visible particle (E px py pz), blank line between events.

#include "Pythia8/Pythia.h"
#include <fstream>
using namespace Pythia8;

int main() {
    const double mH    = 91.188;
    const double Eglue = mH / 2.0;

    Pythia pythia;

    pythia.readString("ProcessLevel:all = off");
    pythia.settings.parm("Beams:eCM", mH);

    // No beam particles in the event, so disable ISR and MPI
    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:MPI = off");

    pythia.readString("Next:numberCount = 1000");

    pythia.init();

    std::ofstream out("../data/Hgg_events.txt");
    const int nEvents = 100000;
    int nWritten = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        pythia.event.reset();

        // Two back-to-back gluons with a color-singlet flow (same as H->gg).
        // Long form: (id, status, m1, m2, d1, d2, col, acol, px, py, pz, e, m)
        // Status 23 = outgoing hard-process particle → gets FSR from Pythia.
        pythia.event.append(21, 23, 0, 0, 0, 0, 101, 102,  0., 0.,  Eglue, Eglue, 0.);
        pythia.event.append(21, 23, 0, 0, 0, 0, 102, 101,  0., 0., -Eglue, Eglue, 0.);

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
