
#ifndef RELOC_DATABASE_HPP_
#define RELOC_DATABASE_HPP_

#include <string>
#include <vector>

namespace FernRelocLib {
class RelocDatabase {
public:
    RelocDatabase(int codeLength, int codeFragmentDim);

    ~RelocDatabase(void);

    /** @return Number of valid similar entries that were found. Mostly
            relevant in case of an empty database.
    */
    int findMostSimilar(
        const char* codeFragments, int nearestNeighbours[], float distances[], int k);

    /** @return ID of newly added entry */
    int addEntry(const char* codeFragments);

    void saveToFile(const std::string& framesFileName) const;

    void loadFromFile(const std::string& filename);

private:
    int mTotalEntries;

    int mCodeLength, mCodeFragmentDim;

    std::vector<int>* mIds;
};
}  // namespace FernRelocLib

#endif
