```mermaid

gitGraph
   commit id: "Pre-Alpha"
   commit id: "Alpha"
        branch feature_1
        commit
        commit
   
   checkout main
   commit id: "Beta"
   branch feature_2
        commit
        branch feature_3
            commit
            commit id: "A"
        checkout feature_2
        merge feature_3
        commit
   checkout main
   merge feature_1
   commit id: "Closed-Beta"
   merge feature_2
   commit id: "Public_Beta"
   branch Bugfixing
        commit
        commit
        cherry-pick id:"A"
   checkout main
    merge Bugfixing
   commit id: "Press Release"
   branch hotfix
        commit
        commit type: REVERSE
   checkout main
   merge hotfix
   commit id: "Version 1.0"