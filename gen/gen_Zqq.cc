// gen_Zqq.cc
// Particle gun: two back-to-back light quarks (d/dbar) at E = mZ/2 each,
// color-singlet configuration matching Z->qqbar. Pythia showers and hadronizes them.
// Symmetric setup to gen_Hgg.cc — only physical difference is quark vs gluon color factor.
//
// Writes 10,000 events to ../data/Zqq_events.txt
// Format: one line per final-state visible particle (E px py pz), blank line between events.

#include "Pythia8/Pythia.h"
#include <fstream>
using namespace Pythia8;

int main() {
    const double mZ   = 91.188;
    const double Equark = mZ / 2.0;

    Pythia pythia;

    pythia.readString("ProcessLevel:all = off");
    pythia.settings.parm("Beams:eCM", mZ);

    pythia.readString("PartonLevel:ISR = off");
    pythia.readString("PartonLevel:MPI = off");

    pythia.readString("Next:numberCount = 1000");

    pythia.init();

    std::ofstream out("../data/Zqq_events.txt");
    const int nEvents = 100000;
    int nWritten = 0;

    for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
        pythia.event.reset();

        // Two back-to-back d/dbar quarks, color-singlet flow (matching Z->qqbar).
        // Long form: (id, status, m1, m2, d1, d2, col, acol, px, py, pz, e, m)
        // Status 23 = outgoing hard-process particle -> gets FSR from Pythia.
        pythia.event.append( 1, 23, 0, 0, 0, 0, 101,   0,  0., 0.,  Equark, Equark, 0.);
        pythia.event.append(-1, 23, 0, 0, 0, 0,   0, 101,  0., 0., -Equark, Equark, 0.);

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
    std::cout << "Wrote " << nWritten << " events to ../data/Zqq_events.txt\n";
    pythia.stat();
    return 0;
}
